from django.db import models

class Clip(models.Model):
    clip_url = models.CharField(max_length=255)
