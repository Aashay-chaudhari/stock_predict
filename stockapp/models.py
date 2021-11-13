from django.db import models

# Create your models here.

class Stock(models.Model):
    symbol = models.CharField(max_length=30)
    lstm1 = models.CharField(max_length=15, default="Neutral")
    lstm2 = models.CharField(max_length=15, default="Neutral")
    lstm1_scope = models.IntegerField(default=0)
    lstm2_scope = models.IntegerField(default=0)




    def __str__(self):
        return self.symbol