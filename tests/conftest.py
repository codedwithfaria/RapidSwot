import sys
import types
from pathlib import Path


def _ensure_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _install_google_adk_stubs() -> None:
    google = _ensure_module("google")
    adk = _ensure_module("google.adk")
    agents = _ensure_module("google.adk.agents")
    events = _ensure_module("google.adk.events")
    invocation = _ensure_module("google.adk.agents.invocation_context")

    class BaseAgent:
        def __init__(self, name: str):
            self.name = name

    class LlmAgent:
        async def generate_response(self, *_: object, **__: object) -> str:
            raise NotImplementedError

    class EventActions:
        def __init__(self, escalate: bool = False):
            self.escalate = escalate

    class Event:
        def __init__(self, author: str, content: object, actions: EventActions):
            self.author = author
            self.content = content
            self.actions = actions

    class InvocationContext:
        class Session:
            def __init__(self) -> None:
                self.state = {}

        def __init__(self) -> None:
            self.session = self.Session()

    agents.BaseAgent = BaseAgent
    agents.LlmAgent = LlmAgent
    events.Event = Event
    events.EventActions = EventActions
    invocation.InvocationContext = InvocationContext

    google.adk = adk
    adk.agents = agents
    adk.events = events
    agents.invocation_context = invocation


def _install_optional_dependency_stubs() -> None:
    _ensure_module("networkx")

    rdflib = _ensure_module("rdflib")

    class Graph:
        def __init__(self) -> None:
            self.triples = []

        def add(self, triple: tuple) -> None:
            self.triples.append(triple)

    class Literal(str):
        pass

    def Namespace(uri: str):
        class _Namespace(str):
            def __getitem__(self, item: str) -> str:
                return f"{uri}{item}"

        return _Namespace(uri)

    rdflib.Graph = Graph
    rdflib.Literal = Literal
    rdflib.Namespace = Namespace


_install_google_adk_stubs()
_install_optional_dependency_stubs()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
