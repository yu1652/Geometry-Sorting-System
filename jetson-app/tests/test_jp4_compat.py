import unittest
from app import exec_qt, qt_enum


class JP4CompatibilityTests(unittest.TestCase):
    def test_qt59_empty_enum_namespace_uses_legacy_zero_value(self):
        class OldQt:
            class EditTrigger:
                pass
            NoEditTriggers = 0
        self.assertEqual(qt_enum(OldQt, 'EditTrigger', 'NoEditTriggers'), 0)

    def test_qt510_dialog_uses_exec_while_retaining_return_code(self):
        class OldQt:
            def exec_(self):
                return 42
        self.assertEqual(exec_qt(OldQt()), 42)

    def test_current_qt_keeps_exec(self):
        class CurrentQt:
            def exec(self):
                return 17
        self.assertEqual(exec_qt(CurrentQt()), 17)


if __name__ == '__main__':
    unittest.main()
