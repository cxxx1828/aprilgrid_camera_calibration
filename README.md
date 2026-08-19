# RealSense D455/D457 AprilGrid Calibration

Kalibracija RGB + levog i desnog IR senzora na Intel RealSense D455/D457 kameri,
korišćenjem AprilGrid table (umesto klasične šahovnice) za veću preciznost detekcije.

## Kalibraciona tabla

- **7 × 10** AprilTag markera
- Rečnik: **tag36h11**
- Veličina taga: **21 mm**
- Razmak između tagova: **6 mm**
- Fabrički generisan PDF za A4 (183 × 264 mm, centrirano, bez skaliranja pri štampi)

> Tabla se štampa na A4 papiru na **100% skali** (isključi "Fit to page" u štampaču),
> inače se fizičke dimenzije neće poklapati sa `TAG_SIZE_MM`/`SPACING_MM` u kodu.

## Struktura projekta

```
aprilgrid_calibration/
├── capture.py          # snimanje sinhronizovanih RGB + IR1 + IR2 slika
├── common.py            # zajedničke helper funkcije (YAML I/O)
├── rgb_mono.py           # mono kalibracija RGB senzora
├── ir_mono.py            # mono kalibracija levog/desnog IR senzora
├── stereo.py             # stereo kalibracija (left-right, rgb-left)
├── calib_images/
│   ├── rgb/
│   ├── left_ir/
│   └── right_ir/
└── results/
    ├── rgb_intrinsics.yaml
    ├── left_ir_intrinsics.yaml
    ├── right_ir_intrinsics.yaml
    ├── stereo_left_right.yaml
    └── stereo_rgb_left.yaml
```

## Zavisnosti

```bash
pip3 install pyrealsense2 opencv-contrib-python numpy
```

`opencv-contrib-python` je obavezan (ne običan `opencv-python`) jer sadrži `cv2.aruco` modul.

## Redosled koraka

### 1. Snimanje slika

```bash
python3 capture.py
```

- `SPACE` — čuva sinhronizovani triplet (RGB + IR1 + IR2), samo ako su sva tri
  frejma detektovala **minimum `MIN_TAGS`** (podrazumevano 20) tagova
- `Q` — izlaz
- IR emiter (laser projektor) se automatski gasi tokom snimanja jer njegov
  tačkasti pattern ometa detekciju tagova
- Preporuka: snimi 150–300 slika sa raznovrsnim uglovima, udaljenostima i
  pozicijama table u kadru (uključujući ivice/uglove slike, ne samo centar)

**Rezolucija/FPS** (`WIDTH, HEIGHT, FPS` na vrhu fajla) mora da odgovara onome
što tvoja USB veza podržava:

- USB3 veza: `1280, 720, 30` radi za sve tri strimove istovremeno
- USB2 veza: kamera ograničava kombinacije (Color max ~15Hz na 720p, IR max ~5Hz);
  proveri `rs-enumerate-devices -c` i po potrebi koristi `640, 480, 30`

> **Kritično**: ne menjaj rezoluciju između snimanja u isti `calib_images/` folder.
> Mešanje rezolucija u istom skupu slika tiho unosi sistemsku grešku u kalibraciju
> (RMS naraste 2–4×) bez ijedne greške pri izvršavanju. Ako menjaš rezoluciju,
> obriši stare slike i snimi ceo set iznova:
> ```bash
> rm -rf calib_images/rgb calib_images/left_ir calib_images/right_ir
> ```

### 2. Mono kalibracija (mora pre stereo koraka)

```bash
python3 rgb_mono.py calibrate
python3 ir_mono.py calibrate left
python3 ir_mono.py calibrate right
```

Svaka komanda:
- detektuje AprilGrid u svakoj slici (`MIN_TAGS` prag)
- računa `cv2.calibrateCamera` (K, distortion coefficients)
- ispisuje RMS reprojection error po celom skupu i po frejmu
- čuva rezultat u `results/*_intrinsics.yaml`

**`stereo.py` čita ove `.yaml` fajlove sa diska** — ne računa intrinsics iznova.
Ako ponovo snimiš slike (novi `capture.py` run), moraš ponovo pokrenuti sve
tri mono komande pre stereo koraka, inače stereo kalibracija koristi zastarele
intrinsics iz starog skupa slika.

### 3. Stereo kalibracija

```bash
python3 stereo.py calibrate left-right
python3 stereo.py calibrate rgb-left
```

- Koristi `cv2.CALIB_FIX_INTRINSIC` — mono intrinsics se ne menjaju, računa se
  samo relativna rotacija/translacija (R, T) između dva senzora
- Za svaki frejm koristi **samo tagove vidljive u oba senzora** (`common_ids`),
  da bi `objpoints`/`imgpoints1`/`imgpoints2` imali isti broj tačaka u istom
  redosledu — obavezno za `cv2.stereoCalibrate`
- Rezultat (K1, dist1, K2, dist2, R, T, E, F) se čuva u `results/stereo_*.yaml`

## Očekivani rezultati (orijentacija)

| RMS reprojection error | Kvalitet |
|---|---|
| < 0.5 px | vrhunski, pogodno za precizan depth/3D |
| 0.5 – 1.5 px | solidno, standard za robotiku (navigacija, AprilTag docking) |
| > 2–3 px | problematično — proveri mešane rezolucije, loš fokus, ili premalu tablu u kadru |

RGB senzor obično ima veći RMS od IR senzora (veća fabrička distorzija), i
`stereo_rgb_left` obično ispadne nešto lošiji od `stereo_left_right` (dva
različita tipa senzora umesto identičnog para) — to je očekivano, ne greška.

## Parametri koje treba menjati zajedno

`GRID_COLS`, `GRID_ROWS`, `TAG_SIZE_MM`, `SPACING_MM`, `TAG_DICT_ID` i
`MIN_TAGS` su definisani identično u `capture.py`, `rgb_mono.py`, `ir_mono.py`
i `stereo.py`. Ako menjaš tablu (npr. štampaš novu na drugoj veličini), ove
vrednosti moraju biti usklađene u sva četiri fajla.

## Debagovanje

- **`Couldn't resolve requests`** pri `capture.py` — tražena kombinacija
  rezolucije/FPS-a nije podržana na trenutnoj USB vezi; proveri
  `rs-enumerate-devices -c` i prilagodi `WIDTH, HEIGHT, FPS`
- **`ModuleNotFoundError: pyrealsense2`** — nije instaliran u aktivnom
  virtualenv-u (`pip3 install pyrealsense2`)
- **Nagli skok u RMS-u posle dodavanja novih slika** — skoro uvek mešane
  rezolucije u istom folderu; obriši i snimi iznova na jednoj rezoluciji
- **`stereo.py` javlja nerealno visok RMS iako su mono rezultati dobri** —
  proveri da li si pokrenula mono kalibracije NA TRENUTNOM skupu slika pre
  stereo koraka (vidi korak 2)
