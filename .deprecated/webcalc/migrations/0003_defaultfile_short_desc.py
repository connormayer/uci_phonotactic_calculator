# Generated by Django 2.1.5 on 2022-05-12 19:33

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("webcalc", "0002_remove_defaultfile_short_desc"),
    ]

    operations = [
        migrations.AddField(
            model_name="defaultfile",
            name="short_desc",
            field=models.CharField(default="", max_length=50),
        ),
    ]
