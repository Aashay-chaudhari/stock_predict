# Generated by Django 3.2.2 on 2021-05-10 08:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stock',
            name='lstm1',
            field=models.CharField(default='Neutral', max_length=15),
        ),
        migrations.AlterField(
            model_name='stock',
            name='lstm2',
            field=models.CharField(default='Neutral', max_length=15),
        ),
    ]
