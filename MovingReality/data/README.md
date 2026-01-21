The data folder has the following structure:
<br>
<br>
- 1019_pp## (folder for participant based on study ID)
	- Vicon (folder with vicon, gold standard, data)<br>
		- .c3d file for each trial.

    		Trialnames have te following configuration: *1019_MR**studyIDnumber**_**trialtype**.c3d*<br>
		  trialtypes: <br>
			1Reg: First trial regular walking<br>
			FBIC: Feedback on initial contact angle<br>
			FBPO: Feedback on push off force<br>
			2FB: Feedback on both initial contact angle and push off force<br>
			2Reg: Second trial regular walking<br>
	- Xsens (folder with IMU sensordata)
 		- corresponding_files.txt: describing which xsens exported folders correspond to which trial and vicon .c3d files 
		- exported###
			- .txt file per sensor.
     				filenames have the following configuration: ***AwindaMasterstationID**_ **trialnumber**_ **SensorID**.txt*
			- sensorspec.json file
     				with specifications of which sensor (SensorID) was attached to which body segment
- ...

The vicon data is saved as *.c3d* files
The xsens sensordata is exported with MT Manager to *.txt( files in the different "exported###" folders for each trial (see "MT Manager export settings.PNG" figure for the export settings).
