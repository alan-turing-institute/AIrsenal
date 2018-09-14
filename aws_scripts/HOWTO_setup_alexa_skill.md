# AIrsenal in AWS

There are a couple of places we can use cloud computing for AIrsenal (using AWS as an example):
 * Store our database (just use an sqlite file on S3, as it is tiny, and we won't have simultaneous read/writes).
 * Update results after the end of a gameweek - run a lambda once per day to see if there is a gameweek that has
 completed but for which data is not yet in the database, and if so, fill them in.
 * Calculate predicted points for the next N gameweeks - this is not too compute-intensive, and
 could potentially be tacked-on to the lambda above - a possible reason not to do this is injury information
 for players that might not be available until nearer the deadline.
 * Optimize transfer strategies for the next N gameweeks.  This is quite compute-intensive - may want to scale out.
 * Use an ***Alexa Skill*** to interact with the code - in particular we can get transfer suggestions (reading from
 the relevant table in the sqlite db), or use the FPL API to get our current score or ranking.

The following notes provide some hints for the last of these - setting up an Alexa skill.

## AIrsenal Alexa skill

### Get AWS developer and Alexa accounts.

### Setup an S3 bucket

 * Go to the [AWS console](https://console.aws.amazon.com/), login, then choose "S3" under the "Storage" heading.
 * Click "Create bucket", give your bucket a unique name, and save it (all default options are fine).

### Setup a Lambda

 * Create a new, empty directory, e.g. ```mkdir ~/zip_airsenal```
 * Create and activate a virtualenv
 ```
 virtualenv airsenal_for_alexa
 source airsenal_for_alexa/bin/activate
 ```
 * From within the virtualenv, install the packages necessary for the lambda:
 ```
 pip install -r /path/to/AIrsenal/aws_scripts/requirements_alexa.txt
 ```
 * Copy the contents of ```site-packages``` for this virtualenv into the empty directory you made earlier:
 ```
 cp -r $VIRTUAL_ENV/lib/python3.6/site-packages/* ~/zip_airsenal
 ```

 * Unfortunately, the Lambda runtime for Python-3.6 doesn't include some of the libraries needed to run our code.
I also needed to unpack and include the contents of the ***sqlite3*** and ***regex*** directories from
[here](https://github.com/Miserlou/lambda-packages) in the ```~/zip_airsenal/``` directory.
 * Finally copy the necessary AIrsenal code into this directory:
 ```
 cp /path/to/AIrsenal/aws_scripts/lambda_airsenal_alexa.py ~/zip_airsenal
 cp -r /path/to/AIrsenal/framework ~/zip_airsenal
 ```
 * Now create the zip file:
 ```
 cd ~/zip_airsenal
 zip -r ~/airsenal.zip .
 ```
 and you should have a file ```airsenal.zip``` in your home directory.  This is what you will upload to AWS as your lambda
 deployment.
 (NOTE - if you make a mistake and need to remake the zip file, remember to delete the old one, otherwise it will get appended-to rather than replaced!)

 * Go to the [AWS console](https://console.aws.amazon.com/) and go to "Lambda" under the "Compute" heading.
 * Click "Create function", then select "Author from scratch", give your function a name, select "Python 3.6" Runtime, "Use an
existing role", and "lambda_basic_execution" in the dropdowns, then click "Create function" at the bottom.
