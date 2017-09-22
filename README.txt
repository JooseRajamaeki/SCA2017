The project in this repository implements the algorithms described in the publication:

Joose Rajamäki, Perttu Hämäläinen, Augmenting Sampling Based Controllers with Machine Learning, SCA 2017


IMPORTANT INFO:

The project works with Visual Studio 2015.

The Visual studio project does not maintain all the commands when transferred. To get the project working, go to the project "SCA" and Properties -> Configuration Properties -> Debugging -> Command and insert "$(ProjectDir)/drawstuffrenderer.exe". You must also change the target platform version to your Windows version. To do that, go to each project's Properties -> Configuration Properties -> General -> Target Platform Version, and change the version to your Windows version. We used Windows 10.0.14393.0.

The main file handling everything is "OdeSimpleWalker.cpp".