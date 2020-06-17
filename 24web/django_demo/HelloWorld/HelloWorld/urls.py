"""HelloWorld URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.urls import re_path
from django.contrib import admin

from . import view, testdb, search, search_post

urlpatterns = [
    url(r'^$', view.hello),
    url('hello/', view.hello),
    re_path(r'^index/$', view.index, name='index'),
    url(r'^testdb$', testdb.testdb),
    re_path('showdb/', testdb.show_db),
    re_path('update_db/', testdb.update_db),
    url(r'^search-form$', search.search_form),
    url(r'^search$', search.search),
    re_path(r'^search-post$', search_post.search_post),
url(r'^admin/', admin.site.urls),

]
