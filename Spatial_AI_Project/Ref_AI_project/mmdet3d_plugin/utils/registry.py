"""
레지스트리 시스템 - mmdet의 레지스트리 시스템을 대체
"""
from collections import defaultdict


class Registry:
    """A registry to map strings to classes.
    
    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))
    """
    
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
    
    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={list(self._module_dict.keys())})'
        return format_str
    
    def __len__(self):
        return len(self._module_dict)
    
    def __contains__(self, key):
        return self.get(key) is not None
    
    def __getitem__(self, key):
        return self._module_dict[key]
    
    def get(self, key):
        """Get the registry record.
        
        Args:
            key (str): The class name in string format.
        
        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)
    
    def register_module(self, name=None, force=False, module=None):
        """Register a module.
        
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        
        Example:
            >>> @BACKBONES.register_module()
            >>> class ResNet:
            >>>     pass
            
            >>> @BACKBONES.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass
            
            >>> BACKBONES.register_module(ResNet)
            >>> BACKBONES.register_module(name='mnet', module=MobileNet)
        
        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        
        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module
        
        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls
        
        return _register
    
    def _register_module(self, module_class, module_name=None, force=False):
        """Register a module.
        
        Args:
            module_class (:class:`type`): Module class to be registered.
            module_name (str | None): The module name. If not specified, the
                class name will be used.
            force (bool): Whether to override an existing class with the same
                name. Default: False.
        """
        if not isinstance(module_class, type):
            raise TypeError(f'module must be a class, but got {type(module_class)}')
        
        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered in {self._name}')
            self._module_dict[name] = module_class
    
    def build(self, cfg, **kwargs):
        """Build a module from config dict.
        
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            **kwargs: Other arguments.
        
        Returns:
            obj: The constructed object.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        if 'type' not in cfg:
            raise KeyError(f'cfg must contain the key "type", but got {cfg}')
        
        args = cfg.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(f'{obj_type} is not in the {self._name} registry')
        else:
            obj_cls = obj_type
        
        return obj_cls(**args, **kwargs)


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    
    Returns:
        obj: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'cfg must contain the key "type", but got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError(f'registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry._name} registry')
    else:
        obj_cls = obj_type
    
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    
    return obj_cls(**args)


# 전역 레지스트리 인스턴스 생성
DETECTORS = Registry('detector')
HEADS = Registry('head')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
BBOX_ASSIGNERS = Registry('bbox_assigner')
BBOX_SAMPLERS = Registry('bbox_sampler')
BBOX_CODERS = Registry('bbox_coder')
MATCH_COST = Registry('match_cost')
PIPELINES = Registry('pipeline')
DATASETS = Registry('dataset')
TRANSFORMER = Registry('transformer')


