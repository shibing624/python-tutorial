from django.contrib import admin

# Register your models here.
from TestModel.models import RiskData,Contact, Tag

admin.site.register(RiskData)
admin.site.register(Tag)
admin.site.register(Contact)