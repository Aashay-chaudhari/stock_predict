# Generated by Django 3.2.2 on 2021-05-10 08:23

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Stock',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(max_length=30)),
                ('lstm1', models.CharField(max_length=15)),
                ('lstm2', models.CharField(max_length=15)),
            ],
        ),
    ]
