"""Tests for the webcalc Django app."""

from django.test import TestCase
from django.urls import reverse

from uci_phonotactic_calculator.cli import legacy


class WebcalcViewTests(TestCase):
    """Test the webcalc views."""

    def test_index_view(self) -> None:
        """Test that the index view returns a 200 status code."""
        response = self.client.get(reverse("webcalc:index"))
        self.assertEqual(response.status_code, 200)


class NgramCalculatorTests(TestCase):
    """Test the integration with the ngram_calculator module."""

    def test_ngram_calculator_import(self) -> None:
        """Test that we can import the ngram_calculator module."""
        self.assertIsNotNone(legacy)
