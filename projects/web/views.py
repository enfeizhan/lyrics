from django.shortcuts import render


# Create your views here.
def index(request):
    if request.GET.get('lyricsInput'):
        lyricsInput = request.GET.get('lyricsInput')
        context = {'output_lyrics': lyricsInput}
    else:
        context = {'output_lyrics': 'You will sing out loud here!'}
    return render(request, 'web/index.html', context)
