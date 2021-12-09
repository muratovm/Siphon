from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def binary(request):
    return render(request, "post.html")
    #return HttpResponse('Hello World')