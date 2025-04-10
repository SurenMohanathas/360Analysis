# 360Analysis - Restaurant Review Analysis Tool

A comprehensive tool for analyzing restaurant reviews from multiple platforms. The application fetches restaurant review data from a PostgreSQL database, performs sentiment analysis, generates insights using LLMs, and creates detailed PDF reports.

## Features

- **Database Integration**: Connect to the PostgreSQL database populated by the "crawler" repository
- **Sentiment Analysis**: Analyze reviews using multiple sentiment analysis techniques
- **LLM-Generated Insights**: Generate comprehensive analysis of reviews using local models or Anthropic's Claude
- **Interactive UI**: Search for restaurants and select date ranges for analysis
- **PDF Report Generation**: Create professional PDF reports with visualizations
- **Trend Analysis**: Analyze sentiment trends over time
- **Command-line Interface**: Run without GUI for automation and scripting

## Requirements

- Python 3.8+
- PostgreSQL database with restaurant reviews (populated by the "crawler" repository)
- Tkinter (for GUI)
- CUDA-compatible GPU (optional, for faster LLM inference)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/360Analysis.git
   cd 360Analysis
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the application:
   ```bash
   cp config.env.example config.env
   # Edit config.env with your database credentials and LLM settings
   ```

## Usage

### GUI Mode

To run the application with the graphical user interface:

```bash
python main.py
```

The GUI allows you to:
- Search for restaurants in the database
- Select a date range for analysis
- Generate a comprehensive PDF report
- View progress and status information

### Command-line Mode

For scripting and automation, the tool can be run from the command line:

```bash
python main.py --no-gui --restaurant "Restaurant Name" --output "./reports" --start-date 2023-01-01 --end-date 2023-12-31
```

Command-line arguments:
- `--no-gui`: Run in command-line mode
- `--restaurant`: Restaurant name to analyze
- `--output`: Output directory for reports
- `--start-date`: Start date for analysis (YYYY-MM-DD)
- `--end-date`: End date for analysis (YYYY-MM-DD)

### Generated Reports

The tool generates PDF reports that include:
- Restaurant information
- Sentiment analysis overview
- Sentiment trends over time
- Strengths and areas for improvement
- Customer experience insights
- Food quality analysis
- Recommendations for improvement
- Sample positive and negative reviews

## LLM Options

The tool supports multiple LLM backends:

1. **Local Models** (default): Uses Hugging Face models locally
   - Supports various open-source models (Mistral, Falcon, Gemma, etc.)
   - Run with CPU or CUDA for GPU acceleration

2. **Hugging Face Models**: Directly uses Hugging Face models
   - Requires a Hugging Face token for some models
   - Better for models too large to run locally

3. **Anthropic Claude**: Uses Anthropic's Claude API
   - Requires an Anthropic API key
   - Provides high-quality analysis but requires internet connection

Configure your preferred LLM in the `config.env` file.

## Project Structure

- `src/database`: Database connection and queries
- `src/analysis`: Sentiment analysis and LLM integration
- `src/report`: PDF report generation
- `src/ui`: User interface components
- `main.py`: Application entry point

## License

This project is licensed under the MIT License - see the LICENSE file for details.