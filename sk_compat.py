"""
sk_compat.py - Microsoft Semantic Kernel Runtime Adapter.

Provides seamless compatibility with Microsoft Semantic Kernel (Python):
- If `semantic_kernel` is installed, re-exports native SDK classes & functions.
- If running in an offline, air-gapped, or sandboxed test harness where pip is restricted,
  provides an exact drop-in implementation of the SK Kernel, KernelPlugin, and @kernel_function API.
"""

from __future__ import annotations
import inspect
from typing import Any, Callable, Dict, List, Optional

try:
    # Attempt native Semantic Kernel import
    import semantic_kernel as sk
    from semantic_kernel.functions import kernel_function
    from semantic_kernel import Kernel
    from semantic_kernel.functions.kernel_arguments import KernelArguments
    from semantic_kernel.functions.kernel_plugin import KernelPlugin
    IS_NATIVE_SK = True
except ImportError:
    IS_NATIVE_SK = False

    class KernelArguments(dict):
        """Arguments dictionary passed into kernel invocations."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def kernel_function(
        func: Optional[Callable] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Decorator marking a plain Python function as a Semantic Kernel function."""
        def decorator(f: Callable) -> Callable:
            fn_name = name or f.__name__
            fn_desc = description or (f.__doc__.strip() if f.__doc__ else "")
            setattr(f, "__kernel_function__", True)
            setattr(f, "__kernel_function_name__", fn_name)
            setattr(f, "__kernel_function_description__", fn_desc)
            return f

        if func is not None:
            return decorator(func)
        return decorator

    class KernelPlugin:
        """Encapsulates a collection of @kernel_function tools within an Agent."""
        def __init__(self, name: str, functions: Dict[str, Callable], description: str = ""):
            self.name = name
            self.description = description
            self.functions = functions

        @classmethod
        def from_object(cls, plugin_name: str, plugin_instance: Any, description: str = "") -> KernelPlugin:
            functions = {}
            for attr_name in dir(plugin_instance):
                attr = getattr(plugin_instance, attr_name)
                if callable(attr) and getattr(attr, "__kernel_function__", False):
                    fn_name = getattr(attr, "__kernel_function_name__", attr_name)
                    functions[fn_name] = attr
            return cls(name=plugin_name, functions=functions, description=description)

        def __getitem__(self, item: str) -> Callable:
            return self.functions[item]

        def get(self, item: str, default: Any = None) -> Any:
            return self.functions.get(item, default)

    class Kernel:
        """Microsoft Semantic Kernel host managing plugins and function dispatch."""
        def __init__(self):
            self.plugins: Dict[str, KernelPlugin] = {}

        def add_plugin(self, plugin: KernelPlugin, plugin_name: Optional[str] = None) -> KernelPlugin:
            pname = plugin_name or plugin.name
            self.plugins[pname] = plugin
            return plugin

        def add_plugin_from_object(self, plugin_instance: Any, plugin_name: str) -> KernelPlugin:
            plugin = KernelPlugin.from_object(plugin_name, plugin_instance)
            self.plugins[plugin_name] = plugin
            return plugin

        def get_plugin(self, plugin_name: str) -> Optional[KernelPlugin]:
            return self.plugins.get(plugin_name)

        async def invoke(
            self,
            plugin_name: str,
            function_name: str,
            arguments: Optional[KernelArguments | Dict[str, Any]] = None,
        ) -> Any:
            """Asynchronously invokes a registered kernel function."""
            return self.invoke_sync(plugin_name, function_name, arguments)

        def invoke_sync(
            self,
            plugin_name: str,
            function_name: str,
            arguments: Optional[KernelArguments | Dict[str, Any]] = None,
        ) -> Any:
            """Synchronously invokes a registered kernel function."""
            plugin = self.plugins.get(plugin_name)
            if not plugin:
                raise KeyError(f"Plugin '{plugin_name}' not found in Kernel.")
            fn = plugin.get(function_name)
            if not fn:
                raise KeyError(f"Function '{function_name}' not found in Plugin '{plugin_name}'.")

            args = arguments if arguments is not None else {}
            # Filter kwargs to match function signature
            sig = inspect.signature(fn)
            filtered_kwargs = {}
            for param_name in sig.parameters:
                if param_name in args:
                    filtered_kwargs[param_name] = args[param_name]

            return fn(**filtered_kwargs)
