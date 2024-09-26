# Democracy Viewer Processing

To use the "No AWS" version of this repository, use the following command to clone the correct branch.

```bash
git clone --single-branch --branch no-aws https://github.com/ryandschaefer/democracy-viewer-processing.git
```

## Instructions to Run

### 1. Install Dependencies (First Time Only)

Run the following command to automatically install all required dependencies.

```bash
sh setup.sh
```

### 2. Create Metadata

Use the template file ```metadata_template.json``` to fill out the metadata for your dataset.

#### data_file

The path to the file that has your dataset. The file must be in a csv format.

#### preprocessing_type

Options to process words in text column(s) of your data. Valid options are ```"none"```, ```"stem"```, and ```"lemma"```. Part of speech tagging is only available if ```"lemma"``` is selected.

#### embeddings

Enable/disable word embeddings for this dataset. Set to ```true``` to enable or ```false``` to disable.

#### embed_col

Grouping column for computing embeddings. Will be ignored if ```embeddings``` is false. **THIS WILL CAUSE AN ERROR IF IT DOES NOT MATCH A VALID COLUMN NAME**.

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

#### text

List of text columns that will be text mined.

#### email

The email address associated with your Democracy Viewer account.

### 3. Run Script

Run the following command to run processing job. 

```bash
python preprocessing.py [metadata_file] [num_threads]
```

Replace ```[metadata_file]``` with the path to the file created in step 2 and replace ```[num_threads]``` with the number of parallel threads you want the program to run with. It is recommended you use no more than the number of cores available on your computer for the ```num_threads``` parameter.

After running this command, status updates will occassionally be printed to the console and an email will be sent to the email associated with your Democracy Viewer account when processing is complete. Depending on your dataset size and processing power, this could take a long time to run.
