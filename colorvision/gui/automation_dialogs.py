"""Dialogs for editing tasks (their steps) and decisions."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..automation import (
    KEYS, STEP_CLICK, STEP_KEY, STEP_LOG, STEP_WAIT, STEP_WAIT_FOR, Decision, Step, Task,
)
from . import theme
from .state_dialog import OPERATOR_LABELS, ConditionRow

STEP_LABELS = {
    STEP_LOG: "Zapisz komunikat",
    STEP_WAIT: "Czekaj",
    STEP_WAIT_FOR: "Czekaj na obiekt",
    STEP_CLICK: "Kliknij obiekt",
    STEP_KEY: "Naciśnij klawisz",
}
BUTTON_LABELS = {"left": "lewym", "right": "prawym"}


class _BaseDialog(QDialog):
    def _finish_layout(self, layout: QVBoxLayout) -> None:
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

    def _fail(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()


class StepRow(QWidget):
    def __init__(self, item_names: list[str], step: Step | None, on_remove, parent=None):
        super().__init__(parent)
        self.kind_combo = QComboBox()
        for kind, label in STEP_LABELS.items():
            self.kind_combo.addItem(label, kind)
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        self.target_combo.setInsertPolicy(QComboBox.NoInsert)
        self.target_combo.addItems(item_names)
        self.button_combo = QComboBox()
        for button, label in BUTTON_LABELS.items():
            self.button_combo.addItem(label, button)
        self.key_combo = QComboBox()
        self.key_combo.addItems(KEYS)
        self.op_combo = QComboBox()
        for op, label in OPERATOR_LABELS.items():
            self.op_combo.addItem(label, op)
        self.value_spin = QSpinBox()
        self.value_spin.setRange(0, 10_000)
        self.value_spin.setValue(1)
        self.value_spin.setFixedWidth(60)
        self.ms_spin = QSpinBox()
        self.ms_spin.setRange(0, 3_600_000)
        self.ms_spin.setSingleStep(100)
        self.ms_spin.setSuffix(" ms")
        self.ms_spin.setValue(1000)
        self.ms_spin.setFixedWidth(100)
        self.message_edit = QLineEdit()
        self.message_edit.setPlaceholderText("Treść komunikatu")
        remove = QPushButton("Usuń")
        remove.clicked.connect(lambda: on_remove(self))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.kind_combo)
        self._fields = (
            self.target_combo, self.op_combo, self.value_spin, self.button_combo, self.key_combo, self.ms_spin, self.message_edit,
        )
        for widget in self._fields:
            layout.addWidget(widget, 1 if widget in (self.target_combo, self.message_edit) else 0)
        layout.addStretch(0)
        layout.addWidget(remove)
        self.kind_combo.currentIndexChanged.connect(self._show_fields)

        if step is not None:
            self.kind_combo.setCurrentIndex(self.kind_combo.findData(step.kind))
            if step.target:
                self.target_combo.setEditText(step.target)
            self.button_combo.setCurrentIndex(self.button_combo.findData(step.button))
            if step.key:
                self.key_combo.setCurrentText(step.key)
            self.op_combo.setCurrentIndex(self.op_combo.findData(step.op))
            self.value_spin.setValue(step.value)
            self.ms_spin.setValue(step.ms)
            self.message_edit.setText(step.message)
        self._show_fields()

    def _show_fields(self) -> None:
        kind = self.kind_combo.currentData()
        visible = {
            STEP_LOG: (self.message_edit,),
            STEP_WAIT: (self.ms_spin,),
            STEP_WAIT_FOR: (self.target_combo, self.op_combo, self.value_spin, self.ms_spin),
            STEP_CLICK: (self.target_combo, self.button_combo),
            STEP_KEY: (self.key_combo,),
        }[kind]
        for widget in self._fields:
            widget.setVisible(widget in visible)

    def step(self) -> Step | None:
        kind = self.kind_combo.currentData()
        target = self.target_combo.currentText().strip()
        if kind in (STEP_CLICK, STEP_WAIT_FOR) and not target:
            return None
        if kind == STEP_LOG:
            return Step(kind, message=self.message_edit.text().strip())
        if kind == STEP_WAIT:
            return Step(kind, ms=self.ms_spin.value())
        if kind == STEP_WAIT_FOR:
            return Step(kind, target=target, ms=self.ms_spin.value(), op=self.op_combo.currentData(), value=self.value_spin.value())
        if kind == STEP_CLICK:
            return Step(kind, target=target, button=self.button_combo.currentData())
        return Step(kind, key=self.key_combo.currentText())


class TaskDialog(_BaseDialog):
    """After accept(), the edited task is available as self.result_task."""

    def __init__(self, task: Task | None, item_names: list[str], taken: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edycja zadania" if task else "Nowe zadanie")
        self.setMinimumWidth(720)
        self.item_names, self.taken = item_names, taken
        self.result_task: Task | None = None
        self.rows: list[StepRow] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        form = QFormLayout()
        self.name_edit = QLineEdit(task.name if task else "")
        self.name_edit.setMaxLength(40)
        form.addRow("Nazwa", self.name_edit)
        layout.addLayout(form)
        title = QLabel("KROKI, WYKONYWANE PO KOLEI")
        title.setProperty("role", "heading")
        layout.addWidget(title)
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(6)
        layout.addLayout(self.rows_layout)
        add = QPushButton("Dodaj krok")
        add.clicked.connect(lambda: self.add_row(None))
        layout.addWidget(add, 0, Qt.AlignLeft)
        hint = QLabel("Kliknięcie celuje w środek największego obiektu danej reguły. Jeśli obiektu nie ma, zadanie kończy się błędem.")
        hint.setProperty("role", "muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._finish_layout(layout)
        for step in (task.steps if task else ()):
            self.add_row(step)
        if not self.rows:
            self.add_row(None)

    def add_row(self, step: Step | None) -> StepRow:
        row = StepRow(self.item_names, step, self.remove_row)
        self.rows.append(row)
        self.rows_layout.addWidget(row)
        return row

    def remove_row(self, row: StepRow) -> None:
        self.rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()

    def accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            self._fail("Podaj nazwę zadania.")
            return
        if name in self.taken:
            self._fail("Zadanie o tej nazwie już istnieje.")
            return
        steps = [row.step() for row in self.rows]
        if not steps or None in steps:
            self._fail("Zadanie musi mieć co najmniej jeden krok, a kliknięcie i oczekiwanie muszą wskazywać obiekt.")
            return
        self.result_task = Task(name, tuple(steps))
        super().accept()


class DecisionDialog(_BaseDialog):
    """After accept(), the edited decision is available as self.result_decision."""

    def __init__(
        self, decision: Decision | None, task_names: list[str], state_names: list[str], item_names: list[str],
        taken: set[str], parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edycja decyzji" if decision else "Nowa decyzja")
        self.setMinimumWidth(620)
        self.item_names, self.taken, self.original = item_names, taken, decision
        self.result_decision: Decision | None = None
        self.rows: list[ConditionRow] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        form = QFormLayout()
        form.setVerticalSpacing(8)
        self.name_edit = QLineEdit(decision.name if decision else "")
        self.name_edit.setMaxLength(40)
        self.task_combo = QComboBox()
        self.task_combo.addItems(task_names)
        self.when_combo = self._state_combo(state_names)
        self.unless_combo = self._state_combo(state_names)
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(-1000, 1000)
        self.priority_spin.setToolTip("Gdy kilka decyzji pasuje naraz, wygrywa najwyższy priorytet")
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 3_600_000)
        self.cooldown_spin.setSingleStep(500)
        self.cooldown_spin.setSuffix(" ms")
        self.cooldown_spin.setValue(5000)
        self.cooldown_spin.setToolTip("Najkrótszy czas między kolejnymi uruchomieniami tej decyzji")
        self.interrupt_box = QCheckBox("Może przerwać zadanie o niższym priorytecie")
        form.addRow("Nazwa", self.name_edit)
        form.addRow("Uruchom zadanie", self.task_combo)
        form.addRow("Gdy stan aktywny", self.when_combo)
        form.addRow("Chyba że aktywny", self.unless_combo)
        form.addRow("Priorytet", self.priority_spin)
        form.addRow("Odstęp", self.cooldown_spin)
        form.addRow("", self.interrupt_box)
        layout.addLayout(form)
        title = QLabel("DODATKOWE WARUNKI, WSZYSTKIE MUSZĄ BYĆ SPEŁNIONE")
        title.setProperty("role", "heading")
        layout.addWidget(title)
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(6)
        layout.addLayout(self.rows_layout)
        add = QPushButton("Dodaj warunek")
        add.clicked.connect(lambda: self.add_row(None))
        layout.addWidget(add, 0, Qt.AlignLeft)
        self._finish_layout(layout)

        if decision is not None:
            self.task_combo.setCurrentText(decision.task)
            self.when_combo.setCurrentIndex(max(0, self.when_combo.findData(decision.when_state)))
            self.unless_combo.setCurrentIndex(max(0, self.unless_combo.findData(decision.unless_state)))
            self.priority_spin.setValue(decision.priority)
            self.cooldown_spin.setValue(decision.cooldown_ms)
            self.interrupt_box.setChecked(decision.interrupt)
            for condition in decision.conditions:
                self.add_row(condition)

    @staticmethod
    def _state_combo(state_names: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItem("(bez warunku)", None)
        for name in state_names:
            combo.addItem(name, name)
        return combo

    def add_row(self, condition) -> ConditionRow:
        row = ConditionRow(self.item_names, condition, self.remove_row)
        self.rows.append(row)
        self.rows_layout.addWidget(row)
        return row

    def remove_row(self, row: ConditionRow) -> None:
        self.rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()

    def accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            self._fail("Podaj nazwę decyzji.")
            return
        if name in self.taken:
            self._fail("Decyzja o tej nazwie już istnieje.")
            return
        if not self.task_combo.currentText():
            self._fail("Najpierw utwórz zadanie.")
            return
        conditions = [row.condition() for row in self.rows]
        if None in conditions:
            self._fail("Każdy warunek musi wskazywać regułę lub wzorzec.")
            return
        when, unless = self.when_combo.currentData(), self.unless_combo.currentData()
        if not (when or unless or conditions):
            self._fail("Wybierz stan albo dodaj warunek, inaczej decyzja nie wiedziałaby, kiedy zadziałać.")
            return
        self.result_decision = Decision(
            name=name,
            task=self.task_combo.currentText(),
            priority=self.priority_spin.value(),
            when_state=when,
            unless_state=unless,
            conditions=tuple(conditions),
            cooldown_ms=self.cooldown_spin.value(),
            interrupt=self.interrupt_box.isChecked(),
            enabled=self.original.enabled if self.original else True,
        )
        super().accept()
