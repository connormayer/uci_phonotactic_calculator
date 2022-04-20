# Generated by Django 2.1.1 on 2022-04-14 20:07

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UploadTrain',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('training_file', models.FileField(upload_to='')),
                ('test_file', models.FileField(upload_to='')),
                ('training_model', models.CharField(choices=[('unigram', 'Unigram Probability'), ('bigram', 'Bigram Probability'), ('posUnigram', 'Positional Unigram Score'), ('posBigram', 'Positional Bigram Score')], max_length=128)),
            ],
        ),
    ]