
# Parking Space Counter

A computer vision project that detects parking space occupancy in a parking lot using Python and OpenCV.

## Features
- Detects parking spaces in a video.
- Classifies spaces as occupied or available using a machine learning model.
- Displays the total, occupied, and available parking spaces.
- Calculates the parking occupancy percentage.
- Allows parking space coordinates to be marked and saved.

## Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-image
- Scikit-learn

## Project Structure
```text
parking-space-counter/
├── main.py
├── mark_parking.py
├── util.py
├── parking_utils.py
├── check_video.py
├── test_model.py
├── model.p
├── parking_spaces.json
├── mask_1920_1080.png
├── requirements.txt
└── samples/
```

## Installation

1. Clone the repository:
   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd parking-space-counter
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download a compatible parking lot video and place it at:
   `samples/parking.mp4`

5. Run the application:
   ```bash
   python main.py
   ```

## Usage

- Run `mark_parking.py` to select parking spaces and save their coordinates.
- Run `main.py` to process the video and view parking occupancy.
- Use `check_video.py` to inspect the input video.
- Use `test_model.py` to test the saved machine learning model.

## Notes
- The video file is excluded from this repository because of its size. Provide your own compatible input video.
- Detection results depend on video quality, camera angle, and model performance.
