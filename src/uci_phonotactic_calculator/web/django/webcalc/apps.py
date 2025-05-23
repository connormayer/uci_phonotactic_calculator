"""Django app configuration for the UCI Phonotactic Calculator web interface."""

from django.apps import AppConfig


class WebcalcConfig(AppConfig):
    """Configuration for the webcalc Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "uci_phonotactic_calculator.web.django.webcalc"
