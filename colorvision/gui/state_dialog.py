"""Dialog for creating and editing a state and its conditions."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..states import MODE_ALL, MODE_ANY, OP_AT_LEAST, OP_AT_MOST, OP_EXACTLY, Condition, StateRule
from . import theme

OPERATOR_LABELS = {OP_AT_LEAST: "co najmniej", OP_AT_MOST: "co najwyżej", OP_EXACTLY: "dokładnie"}
MODE_LABELS = {MODE_ALL: "Wszystkie warunki muszą być spełnione", MODE_ANY: "Wystarczy jeden warunek"}


class ConditionRow(QWidget):
    def __init__(self, item_names: list[str], condition: Condition | None, on_remove, parent=None):
        super().__init__(parent)
        self.item_combo = QComboBox()
        self.item_combo.setEditable(True)  # a plugin can provide names the profile does not know
        self.item_combo.setInsertPolicy(QComboBox.NoInsert)
        self.item_combo.addItems(item_names)
        self.item_combo.setToolTip("Reguła koloru, wzorzec albo nazwa wyniku z pluginu")
        self.op_combo = QComboBox()
        for op, label in OPERATOR_LABELS.items():
            self.op_combo.addItem(label, op)
        self.value_spin = QSpinBox()
        self.value_spin.setRange(0, 10_000)
        self.value_spin.setValue(1)
        self.value_spin.setFixedWidth(70)
        remove = QPushButton("Usuń")
        remove.clicked.connect(lambda: on_remove(self))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.item_combo, 1)
        layout.addWidget(self.op_combo)
        layout.addWidget(self.value_spin)
        layout.addWidget(QLabel("szt."))
        layout.addWidget(remove)
        if condition is not None:
            self.item_combo.setEditText(condition.item)
            self.op_combo.setCurrentIndex(self.op_combo.findData(condition.op))
            self.value_spin.setValue(condition.value)

    def condition(self) -> Condition | None:
        item = self.item_combo.currentText().strip()
        if not item:
            return None
        return Condition(item, self.op_combo.currentData(), self.value_spin.value())


class StateDialog(QDialog):
    """After accept(), the edited state is available as self.result_rule."""

    def __init__(self, rule: StateRule | None, item_names: list[str], taken_names: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edycja stanu" if rule else "Nowy stan")
        self.setMinimumWidth(620)
        self.item_names = item_names
        self.taken_names = taken_names
        self.original = rule
        self.result_rule: StateRule | None = None
        self.rows: list[ConditionRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        self.name_edit = QLineEdit(rule.name if rule else "")
        self.name_edit.setMaxLength(40)
        self.name_edit.setPlaceholderText("np. W walce")
        self.mode_combo = QComboBox()
        for mode, label in MODE_LABELS.items():
            self.mode_combo.addItem(label, mode)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(rule.mode if rule else MODE_ALL))
        self.enter_spin = self._delay_spin(rule.enter_ms if rule else 300, "Tak długo warunek musi być spełniony, aby stan się włączył")
        self.exit_spin = self._delay_spin(rule.exit_ms if rule else 300, "Tak długo warunek musi być niespełniony, aby stan się wyłączył")
        form.addRow("Nazwa", self.name_edit)
        form.addRow("Warunki", self.mode_combo)
        form.addRow("Włącz po", self.enter_spin)
        form.addRow("Wyłącz po", self.exit_spin)
        layout.addLayout(form)

        title = QLabel("WARUNKI")
        title.setProperty("role", "heading")
        layout.addWidget(title)
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(6)
        layout.addLayout(self.rows_layout)
        add = QPushButton("Dodaj warunek")
        add.clicked.connect(lambda: self.add_row(None))
        layout.addWidget(add, 0, Qt.AlignLeft)
        hint = QLabel("„Dokładnie 0 szt.” oznacza, że obiektu nie ma. Wyłączona lub usunięta reguła nigdy nie spełnia warunku.")
        hint.setProperty("role", "muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"color: {theme.DANGER};")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("Zapisz")
        buttons.button(QDialogButtonBox.Cancel).setText("Anuluj")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for condition in (rule.conditions if rule else ()):
            self.add_row(condition)
        if not self.rows:
            self.add_row(None)

    @staticmethod
    def _delay_spin(value: int, tooltip: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 600_000)
        spin.setSingleStep(100)
        spin.setSuffix(" ms")
        spin.setValue(value)
        spin.setToolTip(tooltip)
        return spin

    def add_row(self, condition: Condition | None) -> ConditionRow:
        row = ConditionRow(self.item_names, condition, self.remove_row)
        self.rows.append(row)
        self.rows_layout.addWidget(row)
        return row

    def remove_row(self, row: ConditionRow) -> None:
        self.rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()

    def _fail(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            self._fail("Podaj nazwę stanu.")
            return
        if name in self.taken_names:
            self._fail("Stan o tej nazwie już istnieje.")
            return
        conditions = [row.condition() for row in self.rows]
        if not conditions or None in conditions:
            self._fail("Każdy warunek musi wskazywać regułę lub wzorzec, a stan musi mieć co najmniej jeden warunek.")
            return
        self.result_rule = StateRule(
            name=name,
            conditions=tuple(conditions),
            mode=self.mode_combo.currentData(),
            enter_ms=self.enter_spin.value(),
            exit_ms=self.exit_spin.value(),
            enabled=self.original.enabled if self.original else True,
        )
        super().accept()
