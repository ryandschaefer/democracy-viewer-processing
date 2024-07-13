# Democracy Viewer Processing

## Instructions to Run

### 1. Install Dependencies (First Time Only)

Run the following command to automatically install all required dependencies.

```bash
sh setup.sh
```

### 2. Setup Environment (First Time Only)

Create a file called ```.env``` in the root directory of this repository. Reach out to get the contents of this file.

### 3. Create Metadata

Use the template file ```metadata_template.json``` to fill out the metadata for your dataset.

#### title

The name of your data that all users with access will see. Maximum 30 characters.

#### description

A paragraph description of your data. Maximum 200 characters.

#### author

The original source of the data. Not required, maximum 50 characters.

#### date_collected

The date this dataset was created or collected. Not required.

#### is_public

Set to ```true``` to allow other users to access the data or set to ```false``` so that only you have access. 

#### preprocessing_type

Options to process words in text column(s) of your data. Valid options are ```"none"```, ```"stem"```, and ```"lemma"```. Part of speech tagging is only available if ```"lemma"``` is selected.

#### embeddings

Enable/disable word embeddings for this dataset. Set to ```true``` to enable or ```false``` to disable.

#### embed_col

Grouping column for computing embeddings. Will be ignored if ```embeddings``` was not specified. **THIS WILL CAUSE AN ERROR IF IT DOES NOT MATCH A VALID COLUMN NAME**.

#### language

Langauge the text was written in. Supported languages are listed below.

- Chinese
- English
- French
- German
- Greek
- Italian
- Latin
- Portuguese
- Russian
- Spanish

If the language you are looking for is not currently supported, reach out to see if it is a possibility to add your language.

#### tags

List of tags to help users find your dataset when filtering.

#### text

List of text columns that will be text mined.

#### token

Session token which stores your user data. This allows our server to verify your account and associate this dataset with your account. This token can be found by following these steps:

1. Login to Democracy Viewer.

2. On any page, Right Click -> ```Inspect```.

3. Click on the ```Application``` tab.

4. Under ```Storage```, open ```Local Storage``` and click the URL for Democracy Viewer.

5. Click the row with the key ```democracy-viewer```.

6. Open ```user```. The text inside the quotes after ```token``` is what you need. Make sure to copy everything inside the quotes but not the quotes themselves.

### 4. Run Script

Run the following command to run processing job. 

```bash
python run_pipeline.py [data_file] [metadata_file]
```

Replace ```[data_file]``` with the path to the file with your dataset and replace ```[metadata_file]``` with the path to the file created in step 3.

After running this command, status updates will occassionally be printed to the console and an email will be sent to the email associated with your Democracy Viewer account when processing is complete. Depending on your dataset size and processing power, this could take a long time to run.
