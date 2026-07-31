# -*- coding: utf-8 -*-
"""택소노미 정렬 GT 검출기 — egomotion + obj3d(3DOD)로 클립의 taxonomy 카테고리 예측.

GT로 검출 가능한 카테고리만 산출(나머지 VLM/맥락은 미산출 → 평가에서 별도 표기).
클립 단위 present 집합 + 에피소드별 상세를 반환. gold(사람 라벨) 대비 P/R 평가용.

obj3d: lidar frame x=전방·y=좌우. yaw≈0 동행 / ≈±π 대향 / ≈±π/2 횡단.
       vx,vy 불신 → 위치 미분으로 속도. frame_idx→t 선형(≈10Hz, map ts 미기록).
"""
import numpy as np
import pyarrow.parquet as pq

import paths as P
import events
import classify073 as C

# --- 임계값 ---
LANE_HALF = 2.5      # ego 진행로 반폭(m)
AHEAD = 45.0         # 전방 관심(m)
CUTIN_Y = 3.5        # 차로 밖 판정 |y|(m)
IN_Y = 1.5           # 차로 안 진입 판정 |y|(m)
CLOSE_DIST = 15.0    # 근접 추종 거리(m)
LEAD_DECEL = 1.5     # 선행차 감속 판정(m/s^2, 완만도 포함하도록 확대)
MIN_PTS = 4          # 트랙 최소 관측점
NEAR_CROSS_X = 25.0  # 횡단/교차 상호작용 관심 전방(m)
YIELD_SPEED = 1.0    # 정지 판정 속도(m/s)
EGO_MOVING = 2.0     # ego 진행 판정 속도(m/s)
CREEP_HI = 3.0       # creep 상한 속도(m/s)
DIR_SAME = 0.7       # |yaw|<이 값 → 동행(rad)
DIR_ONCOMING = 2.4   # |yaw|>이 값 → 대향(rad)
# cut_in/out (map 없이 robust): 전방 한정 + 회전 게이트 + 지속 전이
CUTIN_AHEAD = 25.0   # cut_in/out 관심 전방(m) — 원거리 노이즈 제거
ADJ_Y = 2.5          # 인접차로 판정 |y|(m) 하한
ENTER_Y = 1.2        # ego차로 완전 진입 |y|(m)
ATTEMPT_Y = 2.2      # 시도 판정: |y|가 여기까지 감소
ROT_GATE_DEG = 25.0  # 이 이상 ego 회전한 구간에선 cut_in/out 비활성(회전 아티팩트)
SUSTAIN = 3          # 진입/이탈 상태 지속 최소 관측점
HEADWAY_S = 0.6      # close_follow=tailgating: headway 상한(s)
CLOSE_MIN_SPEED = 8.0 # close_follow: ego 최소 속도(m/s, 큐잉 제외)
CREEP_MIN_SEC = 3.0  # creep 최소 지속(s)


def _load_tracks(clip_id, dur):
    of = P.obj3d_frames(clip_id)
    if not of.exists():
        return []
    tp = of.parent / "tracks.parquet"
    if not tp.exists():
        return []
    try:
        cols = set(pq.read_schema(tp).names)
        need = ["track_id", "class_name", "frame_indices", "boxes_3d", "is_observed"]
        if not {"class_name", "frame_indices", "boxes_3d"} <= cols:   # 필수 컬럼 없으면 skip
            return []
        tk = pq.read_table(tp, columns=[c for c in need if c in cols]).to_pydict()
        if "track_id" not in tk:
            tk["track_id"] = list(range(len(tk["class_name"])))
        if "is_observed" not in tk:
            tk["is_observed"] = [[True] * len(b) for b in tk["boxes_3d"]]
    except Exception:
        return []
    # frame_idx 최대값(시간 정규화용) — frames.parquet 기준
    fr = pq.read_table(of, columns=["frame_idx"]).to_pydict()["frame_idx"]
    maxfi = max(fr) or 1
    out = []
    for i in range(len(tk["track_id"])):
        fis = tk["frame_indices"][i]
        bxs = tk["boxes_3d"][i]
        obs = tk["is_observed"][i]
        pts = []
        for fi, b, o in zip(fis, bxs, obs):
            if not o or b is None:
                continue
            pts.append((fi / maxfi * dur, float(b[0]), float(b[1]), float(b[6])))  # t,x,y,yaw
        if len(pts) < MIN_PTS:
            continue
        pts.sort()
        out.append({"cls": tk["class_name"][i], "pts": pts})
    return out


def _agent_speed(t, x, y):
    """위치 미분으로 agent 속도(m/s) 시퀀스."""
    if len(t) < 2:
        return np.zeros_like(t)
    dt = np.gradient(t)
    dt[dt == 0] = 1e-6
    return np.sqrt((np.gradient(x) / dt) ** 2 + (np.gradient(y) / dt) ** 2)


def _ego_speed_at(eg, t0, t1):
    """[t0,t1] 구간 ego 평균/최소 속도."""
    if not eg["ok"]:
        return None, None
    m = (eg["t"] >= t0) & (eg["t"] <= t1)
    if m.sum() == 0:
        return None, None
    return float(eg["speed"][m].mean()), float(eg["speed"][m].min())


def _ego_heading_change(eg, t0, t1):
    """[t0,t1] 구간 ego 순 heading 변화량(deg, 절대) — 회전 게이트용."""
    if not eg["ok"]:
        return 0.0
    m = (eg["t"] >= t0) & (eg["t"] <= t1)
    if m.sum() < 2:
        return 0.0
    yy = eg["yaw"][m]
    return float(abs(np.degrees(yy[-1] - yy[0])))


def detect_taxonomy(clip_id):
    """클립의 GT 검출 가능 taxonomy 카테고리 예측.

    반환: {clip_id, dur, cats:set, detail:[{cat, t0, t1, why}]}
    """
    eg = events.load_egomotion_clip(clip_id)
    ev = events.detect_events(clip_id)
    dur = ev.get("dur", 20.0) if ev.get("ok") else (eg.get("dur", 20.0) if eg["ok"] else 20.0)
    eps = C.consolidate_episodes(ev["events"]) if ev.get("ok") else []

    detail = []

    # ---- egomotion 기동세부 ----
    kinds_all = set(k for e in eps for k in e["kinds"])
    if "lane_change_left" in kinds_all:
        detail.append({"cat": "lane_change_left", "t0": 0, "t1": dur, "why": "egomotion doublet"})
    if "lane_change_right" in kinds_all:
        detail.append({"cat": "lane_change_right", "t0": 0, "t1": dur, "why": "egomotion doublet"})
    # decel_at_intersection: 감속 이벤트가 있고 완전정지로만 끝나지 않으며 ego 진행 중
    for e in ev.get("events", []):
        if e["kind"] == "decelerate":
            am, mn = _ego_speed_at(eg, e["t0"], e["t1"])
            if am and am > EGO_MOVING:
                detail.append({"cat": "decel_at_intersection", "t0": e["t0"], "t1": e["t1"],
                               "why": f"decel, ego mean {am:.1f}m/s"})
                break
    # ---- obj3d 상호작용 ----
    tracks = _load_tracks(clip_id, dur)
    lead_intervals = []                     # 선행차 in-corridor 구간(creep 게이트용)
    for tr in tracks:
        obj = P.OBJ3D_CLASS_MAP.get(tr["cls"], "other")
        pts = tr["pts"]
        t = np.array([p[0] for p in pts]); x = np.array([p[1] for p in pts])
        y = np.array([p[2] for p in pts]); yaw = np.array([p[3] for p in pts])
        ayaw = np.abs(np.arctan2(np.sin(yaw), np.cos(yaw)))  # [0,π]
        mean_dir = float(np.median(ayaw))
        inc = (np.abs(y) < LANE_HALF) & (x > 0) & (x < AHEAD)
        near = (x > 0) & (x < NEAR_CROSS_X) & (np.abs(y) < LANE_HALF + 3)
        aspd = _agent_speed(t, x, y)
        t0i = float(t[inc].min()) if inc.any() else 0.0
        t1i = float(t[inc].max()) if inc.any() else 0.0

        # 보행자/자전거 — 횡단
        if obj in ("pedestrian", "bicycle_micromobility", "motorcycle"):
            crosses = near.any() and (np.abs(y[near]).min() < LANE_HALF) and (np.ptp(y[near]) > 2.0 if near.sum() > 1 else False)
            if near.any() and (obj == "pedestrian"):
                if crosses or np.abs(y[near]).min() < IN_Y:
                    detail.append({"cat": "ped_crossing", "t0": round(t[near].min(), 1),
                                   "t1": round(t[near].max(), 1), "why": "보행자 진행로 횡단"})
                else:   # 근접하나 경로 진입·횡단 아님 → 도로변 보행자(ped_crossing FP 흡수)
                    detail.append({"cat": "vru_roadside", "t0": round(t[near].min(), 1),
                                   "t1": round(t[near].max(), 1),
                                   "why": f"도로변 보행자(min|y|={np.abs(y[near]).min():.1f}m, 비횡단)"})
            elif near.any():
                detail.append({"cat": "cyclist_pm_near", "t0": round(t[near].min(), 1),
                               "t1": round(t[near].max(), 1), "why": f"{obj} 근접"})
            continue

        # 차량류
        if mean_dir < DIR_SAME:            # 동행(선행/후행)
            if not inc.any():
                continue
            # --- cut_in/out: 전방 한정(x>0, x<CUTIN_AHEAD) + 시간순 + 회전 게이트 ---
            fwd = (x > 0) & (x < CUTIN_AHEAD)
            if fwd.sum() >= SUSTAIN:
                ft = t[fwd]; fy = np.abs(y[fwd])           # 이미 시간순 정렬
                fw0, fw1 = float(ft.min()), float(ft.max())
                rot = _ego_heading_change(eg, fw0, fw1)     # 구간 ego 회전량(deg)
                if rot <= ROT_GATE_DEG:                     # 회전 중이면 판정 안 함
                    head = fy[:SUSTAIN].mean()              # 초기 상태
                    tail = fy[-SUSTAIN:].mean()             # 말기 상태
                    ymin = fy.min()
                    if head >= ADJ_Y and tail < ENTER_Y:                 # 인접→차로내 완전 진입
                        detail.append({"cat": "cut_in", "t0": round(fw0, 1), "t1": round(fw1, 1),
                                       "why": f"인접차로({head:.1f}m)→진입({tail:.1f}m)"})
                    elif head >= ADJ_Y and ymin < ATTEMPT_Y:             # 밀고 들어오나 미완
                        detail.append({"cat": "cut_in_attempt", "t0": round(fw0, 1), "t1": round(fw1, 1),
                                       "why": f"인접차로({head:.1f}m)→접근({ymin:.1f}m), 미완"})
                    elif head < ENTER_Y and tail >= ADJ_Y:               # 차로내→이탈
                        detail.append({"cat": "cut_out", "t0": round(fw0, 1), "t1": round(fw1, 1),
                                       "why": f"차로내({head:.1f}m)→이탈({tail:.1f}m)"})
            # --- 선행차: 근접(headway)/급감속 (in-corridor, cut_in/out과 독립) ---
            in_lane_ahead = (np.abs(y[inc]) < IN_Y).mean() > 0.5
            if in_lane_ahead:
                lead_intervals.append((t0i, t1i))    # creep 게이트용
                minx = float(x[inc].min())
                # tailgating = 빠른 주행 중 짧은 headway. 정체/큐잉(저속 근접) 제외.
                egm, _ = _ego_speed_at(eg, t0i, t1i)
                if egm and egm > CLOSE_MIN_SPEED and (minx / egm) < HEADWAY_S:
                    detail.append({"cat": "close_follow", "t0": round(t0i, 1), "t1": round(t1i, 1),
                                   "why": f"tailgating headway {minx/egm:.1f}s ({minx:.0f}m@{egm:.0f}m/s)"})
                if inc.sum() >= 3:
                    va = aspd[inc]
                    if va.max() - va.min() > LEAD_DECEL and va[0] > va[-1]:
                        detail.append({"cat": "lead_decel", "t0": round(t0i, 1), "t1": round(t1i, 1),
                                       "why": "선행차 감속"})
        elif mean_dir > DIR_ONCOMING:      # 대향
            # 대향차가 ego 차로로 침범
            if inc.any() and np.abs(y[inc]).min() < IN_Y:
                detail.append({"cat": "oncoming_encroach", "t0": round(t0i, 1), "t1": round(t1i, 1),
                               "why": "대향차 ego차로 침범"})
        else:                              # 횡단(교차)
            if not near.any():
                continue
            # 교차 차량이 ego 경로 부근에서 정지? ego는? → 우선권 협상
            a_stop = aspd[near].min() < YIELD_SPEED
            tn0, tn1 = float(t[near].min()), float(t[near].max())
            ego_mean, ego_min = _ego_speed_at(eg, tn0, tn1)
            if a_stop and ego_mean is not None and ego_mean > EGO_MOVING:
                detail.append({"cat": "agent_yields_to_ego", "t0": round(tn0, 1), "t1": round(tn1, 1),
                               "why": "교차차 정지·ego 통과"})
            elif ego_min is not None and ego_min < YIELD_SPEED:
                detail.append({"cat": "ego_yields_to_agent", "t0": round(tn0, 1), "t1": round(tn1, 1),
                               "why": "ego 정지·교차차 통과"})

    # ---- creep: 정체 서행 (선행차 존재 ∧ 저속 지속 ∧ 정지/회전 아님) ----
    if eg["ok"]:
        t, sp = eg["t"], eg["speed"]
        stop_iv = [(e["t0"], e["t1"]) for e in ev.get("events", []) if e["kind"] == "stop"]
        turn_iv = [(e["t0"], e["t1"]) for e in ev.get("events", [])
                   if e["kind"] in ("turn_left", "turn_right", "u_turn")]
        creep_mask = (sp > YIELD_SPEED) & (sp < CREEP_HI)
        for i0, i1, a, b in events._runs(creep_mask, t, CREEP_MIN_SEC):
            ov = lambda iv: any(not (b < s or a > e) for s, e in iv)
            # 정지 겹침은 '비율'로 판단: 크롤링 구간이 대부분(>50%) 정지와 겹칠 때만 제외.
            # (기존: 조금이라도 겹치면 제외 → 서행 후 정지하는 정체를 놓침)
            dur_run = max(b - a, 1e-6)
            stop_ov = sum(max(0.0, min(b, e) - max(a, s)) for s, e in stop_iv)
            mostly_stop = stop_ov / dur_run > 0.5
            has_lead = ov(lead_intervals)                 # 앞에 차 있어야 정체
            if has_lead and not mostly_stop and not ov(turn_iv):
                detail.append({"cat": "creep", "t0": round(a, 1), "t1": round(b, 1),
                               "why": f"정체 서행(선행차+저속 {dur_run:.1f}s)"})
                # break 제거: 정체 크롤링이 여러 구간이면 모두 보고(에피소드 정합 위해)

    cats = set(d["cat"] for d in detail)
    return {"clip_id": clip_id, "dur": dur, "cats": cats, "detail": detail}


def cones_in_path(clip_id, dur, w0=None, w1=None, ymax=3.5, xmax=45.0):
    """ego 진행로 내 traffic_cone/bollard 트랙 수 — construction_cones 그라운딩용.

    노변에 흩어진 콘(경로 밖)은 제외하고, ego 경로(|y|<ymax, 0<x<xmax)에 든 콘만 센다.
    실측: 참 공사구간은 경로를 따라 다수 콘이 늘어섬 vs 노변 1~2개는 경로 밖.
    """
    n = 0
    for tr in _load_tracks(clip_id, dur):
        if "cone" not in tr["cls"] and "bollard" not in tr["cls"]:
            continue
        pts = tr["pts"]
        t = np.array([p[0] for p in pts]); x = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts])
        m = (x > 0) & (x < xmax) & (np.abs(y) < ymax)
        if w0 is not None:
            m = m & (t >= w0 - 1.0) & (t <= w1 + 1.0)
        if m.any():
            n += 1
    return n


# GT 검출 가능 카테고리(평가 대상) / VLM·맥락 전용(미산출)
GT_CATS = ["cut_in", "cut_in_attempt", "cut_out", "lead_decel", "close_follow", "ped_crossing",
           "vru_roadside", "cyclist_pm_near", "oncoming_encroach", "agent_yields_to_ego",
           "ego_yields_to_agent", "lane_change_left", "lane_change_right", "decel_at_intersection", "creep"]
VLM_CATS = ["intersection_signalized", "intersection_unsignalized", "roundabout", "merge_onramp",
            "construction_cones", "toll_gate", "red_light_stop", "signal_go", "pull_over",
            "road_highway", "road_urban_arterial", "road_backstreet", "road_rural",
            "road_tunnel", "road_bridge", "road_parking"]
