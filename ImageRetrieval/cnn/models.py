from django.db import models

class ImageTest(models.Model):
    image = models.ImageField(upload_to='', blank=True, null=True)
