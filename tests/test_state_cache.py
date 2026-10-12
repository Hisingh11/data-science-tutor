import unittest
from types import SimpleNamespace
from unittest.mock import patch

import ui.state as app_state


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class CoreCacheTests(unittest.TestCase):
    def test_existing_session_core_rebuilds_for_new_model_cache_version(self):
        old_core = SimpleNamespace(model=object())
        session_state = _SessionState({
            "core": old_core,
            app_state.CORE_MODEL_VERSION_KEY: "old-version",
        })
        model = object()
        rag = object()
        with (
            patch.object(app_state.st, "session_state", session_state),
            patch.object(app_state, "shared_model", return_value=model) as shared_model,
            patch.object(app_state, "shared_rag", return_value=rag),
            patch.object(app_state, "SelfRAGEngine"),
            patch.object(app_state, "InterviewSystem"),
            patch.object(app_state, "AssignmentManager"),
            patch.object(app_state, "CodeAssistant"),
            patch.object(app_state, "DeepResearchEngine"),
        ):
            refreshed_core = app_state.core()
            second_core = app_state.core()

        self.assertIsNot(refreshed_core, old_core)
        self.assertIs(refreshed_core, second_core)
        self.assertIs(refreshed_core.model, model)
        shared_model.assert_called_once_with(app_state.MODEL_CACHE_VERSION)

    def test_reset_core_clears_session_model_version(self):
        session_state = _SessionState({
            "core": object(),
            app_state.CORE_MODEL_VERSION_KEY: app_state.MODEL_CACHE_VERSION,
        })
        with (
            patch.object(app_state.st, "session_state", session_state),
            patch.object(app_state.shared_model, "clear") as clear_cache,
        ):
            app_state.reset_core()

        self.assertNotIn("core", session_state)
        self.assertNotIn(app_state.CORE_MODEL_VERSION_KEY, session_state)
        clear_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
