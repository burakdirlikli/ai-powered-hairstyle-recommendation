# AI Hairstyle Recommendation Engine

> ⚠️ **Work In Progress (WIP)**
> This project is currently under active development. Some important features are still missing or incomplete. Specifically, the integration of the recommendation system with the `virtual_try_on` module is not yet functional.

This project is an AI-powered recommendation engine that analyzes a user's face shape and current hair type to suggest the most suitable hairstyles based on personal preferences (hair length, maintenance, beard status, etc.).

It is designed as an isolated and modular "AI Engine," utilizing OpenAI's Vision capabilities alongside traditional machine learning approaches.

## Features

- **Face & Hair Analysis:** Predicts the user's face shape (oval, square, round, rectangular) and hair type (straight, wavy/curly, buzz) from a given photo.
- **Candidate Selection:** Filters the best matching candidates from a database (CSV) of hairstyles based on the initial analysis.
- **Smart Recommendation System:** Uses OpenAI Vision API (GPT-4o Vision) to recommend the top 5 hairstyles by taking the user's personal preferences into account.
- **Virtual Try-On (WIP):** *Currently under development.* The infrastructure is being built to show how the selected hairstyles will look on the user's photo.

## Prerequisites

- Python 3.8+
- [OpenAI API Key](https://platform.openai.com/)

## Installation

1. **Clone the Repository:**
   ```bash
   cd recommendation-system
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # For Windows:
   venv\Scripts\activate
   # For macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables:**
   Open the `.env` file located in the project directory with a text editor and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-your-api-key-here
   ```

## Usage

You can run the system via the command line. A user photo is required to run the engine.

To run with default settings:
```bash
python engine_runner.py --image "samples/test_image.jpg"
```

To run with specific user preferences, you can provide a JSON file (e.g., `prefs.json`):
```bash
python engine_runner.py --image "samples/test_image.jpg" --prefs "samples/prefs.json"
```

### Example Preferences JSON Format
```json
{
    "hair_length": "medium",
    "maintenance": "low",
    "beard": "yes",
    "usage": "casual"
}
```

## Project Structure

- `engine_runner.py`: The main orchestration script. It combines the analysis, selection, and recommendation steps.
- `ml_service/`: Directory containing machine learning models, prediction functions, and AI business logic.
- `data/`: Contains the hairstyle database (`hs_definitions.csv`) and related metadata.
- `requirements.txt`: Lists project dependencies.
- `test_genai_run.py`: Utility script used for testing Generative AI (GenAI) components.

## License

This project is developed for personal/educational purposes. All rights reserved.