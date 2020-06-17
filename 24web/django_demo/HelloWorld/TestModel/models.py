# Create your models here.
from django.db import models


class RiskData(models.Model):
    userid = models.CharField(max_length=20)
    ideaid = models.CharField(max_length=20)
    query = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    desc1 = models.CharField(max_length=255)
    show_url = models.CharField(max_length=255)
    target_url = models.CharField(max_length=255)
    show_date = models.CharField(max_length=255)
    risk_name = models.CharField(max_length=255)
    risk_word = models.CharField(max_length=255)


class Contact(models.Model):
    name = models.CharField(max_length=200)
    age = models.IntegerField(default=0)
    email = models.EmailField()

    def __unicode__(self):
        return self.name


class Tag(models.Model):
    contact = models.ForeignKey(Contact, on_delete=models.CASCADE, )
    name = models.CharField(max_length=50)

    def __unicode__(self):
        return self.name