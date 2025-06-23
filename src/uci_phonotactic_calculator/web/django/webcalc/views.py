"""Views for the webcalc Django app."""

from os import makedirs
from os.path import basename, dirname, exists, join

from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView
from django.views.generic.edit import CreateView

from uci_phonotactic_calculator.utils import utility as util
from uci_phonotactic_calculator.cli.legacy import run
from uci_phonotactic_calculator.web.django import settings

from .models import DefaultFile, UploadTrain, UploadWithDefault


def index(request):
    """Redirect to the home page."""
    return render(request, "webcalc/home.html")


class UploadTrainView(CreateView):
    """View for uploading training and test files."""

    model = UploadTrain
    fields = ["training_file", "default_training_file", "test_file"]
    template_name = "webcalc/home.html"
    success_url = reverse_lazy("webcalc:output")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["documents"] = self.model.objects.all()
        return context

    def form_invalid(self, form):
        response = super(UploadTrainView, self).form_invalid(form)
        context = self.get_context_data()
        form.data = form.data.copy()  # make copy of form data
        form.data["default_training_file"] = ""
        context["form"] = form  # set form in context to updated form
        return response

    def form_valid(self, form):
        response = super(UploadTrainView, self).form_valid(form)
        objects = self.model.objects.last()

        media_path = settings.MEDIA_ROOT

        uploaded_training = objects.training_file.name != ""
        default_training = objects.default_training_file != ""

        # This may not be needed
        if uploaded_training and default_training:
            messages.warning(
                self.request,
                "Either upload a training file OR use a default one (not both)",
            )
            return self.form_invalid(form)

        if not uploaded_training:
            if not default_training:
                messages.warning(
                    self.request,
                    "Please upload a training file or select a default file",
                )
                return self.form_invalid(form)
            else:
                train_file = join(
                    media_path, "default", basename(objects.default_training_file)
                )
        else:
            train_file = join(
                media_path, "uploads", basename(objects.training_file.name)
            )

        test_file = join(media_path, "uploads", basename(objects.test_file.name))

        # Validate training and test files
        train_success, message = util.valid_file(train_file)
        if not train_success:
            messages.warning(self.request, f"Invalid training file format: {message}")
            return self.form_invalid(form)

        test_success, message = util.valid_file(test_file)
        if not test_success:
            messages.warning(self.request, f"Invalid test file format: {message}")
            return self.form_invalid(form)

        outfile_name = util.get_filename(
            basename(objects.test_file.name), objects.current_time.timestamp()
        )
        directory = join(media_path, "uploads", dirname(outfile_name))

        if not exists(directory):
            makedirs(directory)

        out_file = join(media_path, "uploads", outfile_name)

        run(train_file, test_file, out_file)

        return response


class MediaView(TemplateView):
    """View for displaying available datasets."""

    template_name = "webcalc/media.html"
    model = DefaultFile

    def get_context_data(self, **kwargs):
        # Ensure default dataset entries exist so the list is never empty.
        if self.model.objects.count() == 0:
            from pathlib import Path

            default_dir = Path(settings.MEDIA_ROOT) / "default"
            if default_dir.exists():
                # mapping from UploadTrain helper
                try:
                    from .models import UploadTrain  # local import to avoid circularity

                    desc_map = dict(UploadTrain.get_file_choices())
                except Exception:
                    desc_map = {}
                for fp in default_dir.iterdir():
                    if fp.is_file():
                        # Avoid duplicates if a race condition occurs
                        self.model.objects.get_or_create(
                            training_file=f"default/{fp.name}",
                            defaults={
                                "description": desc_map.get(fp.name, fp.stem.replace("_", " ").title()),
                                "short_desc": desc_map.get(fp.name, fp.stem)[:50],
                            },
                        )
        context = super().get_context_data(**kwargs)
        context["objects"] = self.model.objects.all()
        return context


class OutputView(TemplateView):
    """View for displaying calculation results."""

    model = UploadTrain
    template_name = "webcalc/output.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        objects = self.model.objects.last()
        file_name = util.get_filename(
            basename(objects.test_file.name), objects.current_time.timestamp()
        )
        context["output_file"] = file_name
        context["human_readable_output_file"] = basename(file_name)
        return context


class AboutView(TemplateView):
    """View for displaying information about the calculator."""

    template_name = "webcalc/about.html"


class UploadDefaultView(CreateView):
    """View for uploading with default training files."""

    model = UploadWithDefault
    fields = ["training_file", "test_file"]
    template_name = "webcalc/uploadDefault.html"
    success_url = reverse_lazy("webcalc:output")

    def form_invalid(self, form):
        response = super(UploadDefaultView, self).form_invalid(form)
        context = self.get_context_data()
        form.data = form.data.copy()  # make copy of form data
        form.data["training_file"] = (
            ""  # reset training file dropdown selection in form
        )
        context["form"] = form  # set form in context to updated form
        messages.warning(self.request, "Bad file formatting")
        return response

    def form_valid(self, form):
        response = super(UploadDefaultView, self).form_valid(form)
        objects = self.model.objects.last()

        media_path = settings.MEDIA_ROOT

        train_file = join(media_path, "default", objects.training_file)
        test_file = join(media_path, "uploads", basename(objects.test_file.name))

        # Validate test file
        test_success, message = util.valid_file(test_file)
        if not test_success:
            messages.warning(self.request, f"Invalid test file format: {message}")
            return self.form_invalid(form)

        outfile_name = objects.out_file
        out_file = join(media_path, "uploads", outfile_name)

        run(train_file, test_file, out_file)

        return response
