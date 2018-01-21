from django.db import models


# Create your models here.
class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.CharField(max_length=50)
    signup_datetime = models.DateTimeField()

    def __str__(self):
        return self.first_name + ' ' + self.last_name


class Lyrics(models.Model):
    lyrics_id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lyrics_text = models.CharField(max_length=10000)

    def __str__(self):
        return self.lyrics_text
