# UCI Phonotactic Calculator Web Interface

This directory contains a Django web application for the UCI Phonotactic Calculator.

## Installation

The web interface is provided as an optional component of the package. To install it, use:

```bash
pip install 'uci-phonotactic-calculator[web]'
```

This will install Django and other dependencies required for the web interface.

## Running the Web Interface

After installation, you can run the web interface using:

```bash
uci-phonotactic-web runserver
```

This will start a development server at http://127.0.0.1:8000/ where you can access the web interface.

## Development

If you're developing the web interface, you can run the Django management commands directly:

```bash
python -m uci_phonotactic_calculator.web.django.manage runserver
```

## Structure

The web interface follows a standard Django application structure:

- `manage.py` - Django management script
- `settings.py` - Django settings
- `urls.py` - URL routing configuration
- `wsgi.py` - WSGI application entry point
- `webcalc/` - The main Django app containing models, views, etc.

## Testing

You can run tests for the web interface using:

```bash
python -m uci_phonotactic_calculator.web.django.manage test uci_phonotactic_calculator.web.django.webcalc
```
