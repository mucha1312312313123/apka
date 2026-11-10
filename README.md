# OSRS Color Vision

Wykrywanie obiektów po kolorze i wyglądzie w oknie zamkniętego klienta. Program czyta piksele z ekranu: nie modyfikuje klienta i nie wstrzykuje kodu. Myszą i klawiaturą steruje wyłącznie po ręcznym włączeniu trybu **Prawdziwe sterowanie**, przeznaczonego do wersji offline.

## Wersja .exe

Plik `dist/OSRS Color Vision.exe` nie wymaga Pythona ani instalacji: skopiuj go na inny komputer z 64-bitowym Windows i kliknij dwukrotnie.

- Profile, pluginy i ustawienia są w `%APPDATA%\OSRS Color Vision`. Nowa wersja `.exe` ich nie nadpisuje, więc zmiany w domyślnym profilu z nowej wersji nie pojawią się same.
- Pierwsze uruchomienie zwykle trwa dłużej, bo plik rozpakowuje się do folderu tymczasowego.
- Plik nie jest podpisany, więc Windows może pokazać ostrzeżenie SmartScreen: **Więcej informacji → Uruchom mimo to**.
- Własne pluginy mogą importować tylko to, co jest w pliku: bibliotekę standardową Pythona (o ile została dołączona), `cv2`, `numpy` i `colorvision`.

Po zmianie kodu zbuduj plik ponownie, uruchamiając `build_exe.bat`.

## Instalacja ze źródeł

1. Skopiuj cały folder na drugi komputer.
2. Zainstaluj Python 3.10 lub nowszy.
3. Kliknij dwukrotnie `install.bat` (potrzebny internet) i poczekaj na napis `GOTOWE`. Tworzenie środowiska może potrwać do minuty.

## Uruchomienie

Kliknij dwukrotnie `run.bat`. Otworzy się okno programu:

1. Wybierz okno klienta z listy u góry i naciśnij **Start** (F5).
2. Wybierz regułę po lewej, włącz **Pipetę** (F7) i kliknij kolor na obrazie.
3. Popraw zakres suwakami H, S i V. **Maska** (F8) pokazuje dokładnie, co pasuje do reguły.
4. Zapisz profil (Ctrl+S).

Jeśli program łapie cały ekran albo złe okno:

- naciśnij **Wskaż okno** (F2) i w ciągu 3 sekund kliknij grę,
- gdy gra jest zmaksymalizowana albo pełnoekranowa, po **Start** naciśnij **Obszar** (F3) i przeciągnij myszą po jej fragmencie. **Cały obraz** cofa ten wybór.

Na dole okna widać tytuł i rozmiar tego, co jest przechwytywane.

Skróty: F1 pomoc, F2 Wskaż okno, F3 Obszar, F5 Start/Stop, F6 Pauza, F7 Pipeta, F8 Maska, F9 zapis klatki, F10 wycinanie wzorca, F11 Debug. Ustawienia okna są w pliku `settings.ini`, a nieoczekiwane błędy trafiają do `colorvision_error.log`.

Program jest też dostępny z wiersza poleceń:

```
run.bat windows
run.bat run --title "Old School RuneScape"
```

Polecenia są dla `cmd`. W PowerShellu pisz `.\run.bat` zamiast `run.bat`.

`windows` wypisuje tytuły otwartych okien. Jeśli klient offline nazywa się inaczej, podaj jego tytuł w `--title`.

Okno klienta musi być widoczne i nieprzykryte, bo program odczytuje obraz z ekranu. Okno podglądu ustaw obok klienta.

## Kalibracja kolorów

Domyślne zakresy w `profiles/default.json` to tylko punkt startowy, więc skalibruj je na swoim kliencie:

```
run.bat calibrate --rule hp_bar_green --region game_view
```

1. Kliknij kolor na obrazie.
2. Popraw zakres suwakami. Pasujące piksele zostają kolorowe, a reszta jest przyciemniona.
3. Naciśnij `S`, żeby zapisać, i `Q`, żeby wyjść.

Podanie nowej nazwy w `--rule` tworzy nową regułę. Przy oknie w ruchu wygodniej jest kalibrować na stopklatce:

```
run.bat snapshot -o klatka.png
run.bat calibrate --image klatka.png --rule moj_obiekt
```

## Rozpoznawanie kształtów i rodzajów

**Kształt** ustawiasz w regule koloru: Dowolny, Pasek, Prostokąt, Koło, Kropka albo Inny. Reguła zgłasza wtedy tylko obiekty o tym kształcie, a rozpoznany kształt widać w zakładce **Wykrycia**. Kształt nie mówi, czym jest obiekt, ale odrzuca inne rzeczy w tym samym kolorze.

**Wzorzec** rozpoznaje konkretną rzecz po wyglądzie:

1. Naciśnij **Wytnij wzorzec** (F10). Obraz zatrzyma się na czas zaznaczania.
2. Przeciągnij myszą po obiekcie, jak najciaśniej wokół niego.
3. Nadaj wzorcowi nazwę i ustaw wymagane podobieństwo.

Wzorce najlepiej działają na elementach o stałym wyglądzie, np. przedmiotach, ikonach i przyciskach. Modele 3D zmieniają wygląd przy obracaniu, więc mogą nie zostać znalezione. Wzorce zapisują się w pliku profilu, więc kopiując profil, kopiujesz też je.

## Stany

Zakładka **Stany** śledzi, co dzieje się w czasie, np. czy trwa walka i od kiedy. Stany są od siebie niezależne, więc kilka może być aktywnych naraz.

- **Warunek** określa, ile obiektów ma zgłaszać reguła koloru, wzorzec lub plugin, np. `hp_bar_green: co najmniej 1`. `dokładnie 0` oznacza, że obiektu nie ma.
- **Włącz po / Wyłącz po** to czas, przez jaki warunek musi się utrzymać. Dzięki temu jedna błędna klatka nie przełączy stanu.
- Wyłączona lub usunięta reguła nigdy nie spełnia warunku, a w tabeli pojawia się ostrzeżenie `⚠ brak`.

Domyślny profil ma dwa przykłady. **W walce** działa od razu, ale pasek HP innej postaci w pobliżu też go włączy. **Ekwipunek pełny** wymaga wzorca o nazwie `pusty_slot`: wytnij pusty slot ekwipunku i nadaj mu tę nazwę. Stan włączy się, gdy nie będzie żadnego pustego slotu.

## Pewność, obiekty, zdarzenia i historia

- **Pewność** mówi, jak dobrze obiekt pasuje do reguły, a nie jak prawdopodobne jest, że to właściwa rzecz. Dla koloru łączy udział naprawdę pasujących pikseli i bliskość odcienia do środka zakresu. Dla wzorca jest to podobieństwo obrazu. W regule koloru można ustawić **Min. pewność**.
- **Obiekty** nadają wykryciom stałe numery między klatkami i liczą prędkość. Dopasowanie opiera się na odległości, więc dwa bliskie, mijające się obiekty tej samej reguły mogą zamienić się numerami. Numery są ważne tylko do następnego Startu.
- **Zdarzenia** to pojawienie się, zniknięcie, ruch, zatrzymanie oraz początek i koniec stanu. Obiekt widoczny tylko przez jedną klatkę nie tworzy zdarzenia.
- **Historia** pokazuje statystyki i wykres z ostatniej minuty. Trzyma do około 10 minut próbek na regułę, wyłącznie w pamięci, więc przed zamknięciem programu wyeksportuj ją do CSV. Pliki są rozdzielane średnikami pod polski Excel.
- **Debug** (F11) rysuje na podglądzie numery obiektów, pewność, ślady ruchu, regiony, stany i czasy przetwarzania. Działa także na zapisanych klatkach i zatrzymanym obrazie.

## Automatyzacja (tylko wersja offline)

Na serwerach Jagex automatyzacja jest zabroniona i grozi banem. Program nie ma mechanizmów jej ukrywania: klika w stałe punkty, bez losowych opóźnień i bez „ludzkiego” ruchu myszy.

Zakładka **Automat** składa się z trzech części:

- **Decyzje** wybierają, co zrobić: gdy spełnione są stan i warunki, uruchamiają zadanie. Przy kilku pasujących wygrywa najwyższy priorytet.
- **Zadania** to kolejne kroki: zapisz komunikat, czekaj, czekaj na obiekt, kliknij obiekt, naciśnij klawisz. Działa jedno zadanie naraz.
- **Akcje** wykonują kliknięcia i klawisze oraz pilnują zasad bezpieczeństwa.

Zawsze zaczynaj od trybu **Symulacja**. Pokazuje on w zakładce **Zdarzenia**, co program by zrobił, niczego nie klikając.

Zabezpieczenia trybu **Prawdziwe sterowanie**:

- po każdym uruchomieniu i po każdym Stop tryb wraca do **Wyłączona**, a włączenie wymaga potwierdzenia,
- **F12** zatrzymuje sterowanie natychmiast, także gdy aktywne jest okno gry,
- akcja wykona się tylko wtedy, gdy okno klienta jest aktywne i nieprzykryte, a cel leży w jego wnętrzu,
- pauza, zniknięcie okna lub zmiana trybu przerywa bieżące zadanie,
- najwyżej jedna akcja na klatkę, co najmniej 100 ms przerwy, a po 500 akcjach sterowanie samo się wyłącza.

## Regiony

Regiony w profilu są opisane dla układu **Fixed** o rozmiarze 765x503 i mają przybliżone wymiary. Przy większym oknie są skalowane proporcjonalnie, co pasuje tylko do rozciągniętego układu Fixed. W układzie Resizable elementy interfejsu leżą gdzie indziej, więc popraw `regions` albo usuń `region` z reguły (`"region": null`).

## Pisanie pluginów

Dodaj plik `.py` do folderu `plugins/`. Pliki zaczynające się od `_` są pomijane.

```python
from colorvision import Plugin

class MojPlugin(Plugin):
    name = "moj_plugin"

    def on_frame(self, result):
        cel = result.best("hp_bar_green")   # największy obiekt albo None
        if cel:
            print(cel.center, cel.area)

    def on_state_change(self, change):
        print(change.name, "aktywny" if change.active else "nieaktywny")

    def draw(self, canvas, result):
        pass  # rysowanie w oknie podglądu (OpenCV)
```

Metoda `on_event(event)` dostaje wszystkie zdarzenia, a `result.tracks` zawiera śledzone obiekty. W `on_frame` możesz też sprawdzić stan przez `result.is_active("W walce")`. W tej metodzie stany pochodzą z poprzedniej klatki, bo wyniki pluginów też mogą być warunkami.

Plugin może też dopisać własne wyniki, np. z modelu AI, do `result.detections["nazwa"]` w `on_frame`. Trafią one do zakładki **Wykrycia** i będą widoczne dla kolejnych pluginów, a ramki rysuje wtedy sam plugin w `draw`.

Pluginy włączasz i wyłączasz w zakładce **Pluginy**, a ich komunikaty `print()` trafiają do zakładki **Dziennik**. Plugin, który zgłosi wyjątek, zostaje wyłączony, a reszta działa dalej. Przykład znajdziesz w `plugins/target_watch.py`.

## Testy

```
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest tests
```
