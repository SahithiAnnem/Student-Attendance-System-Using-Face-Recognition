
from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView

from attendance.views import login, registration, logout, activate_account, addfaculty, getfacultys, deletefaculty, \
    activateAccount, getstudents, deletestudent, addinattendance, addoutattendance, getattendance, loginpage

urlpatterns = [

    path('admin/', admin.site.urls),

    path('',TemplateView.as_view(template_name = 'index.html'),name='login'),
    path('login/',loginpage,name='login'),
    path('loginaction/',login,name='loginaction'),
    path('registration/',TemplateView.as_view(template_name = 'registration.html'),name='registration'),
    path('regaction/',registration,name='regaction'),
    path('activate/', TemplateView.as_view(template_name='activate.html'), name='registration'),
    path('activateaction/', activate_account, name='regaction'),
    path('logout/', logout, name='logout'),

    path('activatestudent/',activateAccount,name='activateAccount'),
    path('getstudents/',getstudents,name='regaction'),
    path('deletestudent/',deletestudent,name='regaction'),

    path('addfaculty/',TemplateView.as_view(template_name ='addfaculty.html'),name='apply'),
    path('addfacultyaction/',addfaculty,name='add'),
    path('getfacultyes/',getfacultys,name='view'),
    path('deletefaculty/',deletefaculty,name='delete'),

    path('attendancein/',addinattendance,name='add'),
    path('attendanceout/',addoutattendance,name='view'),
    path('viewattendance/',getattendance,name='delete'),

]
