# Digital Biopsy - Deep Segmentation
This is the segmentation tool for digital biopsy project. It also contains some useful git command instructions for the reference.
## To Do List
- [x] Randomize the train test split.
- [x] Add data (input/target/test) into .gitignore.
- [ ] Test if current model work on cpu devices.
- [x] Train test split already included in the code, delete the validation folder.
- [x] Intergrate the image tiling with the training script.
- [ ] Refactor code to enable training on server/local pc.
- [ ] Check the optimizer logic
- [ ] Add grayscale augmentation methods to prevent overfitting on color.
- [x] Move parameters to 'params_meta.py'
## To Run The Script
#### Commands
Data pre-processing is required before training. The data path is defined in 'params-meta.py'
```
python main.py -prep
```
Hyperparameters are defined in 'params-meta.py'
```
python main.py -train
```
The following command will predict using the trained model and save the predictions under the 'pred' folder<br>
Note: the model must align with the weight matrices that used to predict. Changing the hyperparameters might cause runtime failures.
```
python main.py -pred
```
#### Optional command(s)
The following optional command works with '-pred' and '-train'. This will verbosely print out some of the training progresses that help debugging.
```
-v
```
## Git Commands
The easiest way to use github is through [Github Desktop](https://desktop.github.com/). But here are some commands that might be helpful in case of necessary.
#### 1. Clone the repository
```
git clone <http address of the repository>
```
```
git clone https://github.com/digital-biopsy/deep-segmentation.git
```
Github might prompt user to enter their token before access the repository. Setup instructions are [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). Don't forget to take down the token somewhere.
#### 2. Create a branch
Creating a branch helps users to commit changes that haven't been fully tested.
1. Create a branch in local repository.
```
git branch checkout -b <your branch name>
```
```
git branch checkout -b dependency-fix
```
2. Push to the remote branch
Creating a branch locally would not chage the remote repository. The below command can push the new local branch to remote.
```
git push -u origin <a branch that not in the remote>
```
3. Switch to other branch.
All changes in the current branch need either be commited or stashed.
```
git checkout <an already existed branch>
```
#### 3. Git Commit and Stash
Git commit allow you to commit changes locally which git push save changes remotely.
1. Check current changes
git status will print out all the changed files.
```
git status
```
2. Git add
git add will add all changes but sometime its not desirable because it might also add dependencies or cache files.
```
git add .
```
```
git add <file path in the root directory and the file name>
```
3. Git commited
```
git commit -m "some commit message about the change"
```

4. Git Push
Push changes to remote branch. For the most of the time, remote names are just 'origin'
```
git push <remote name> <branch name>
```
5. Git pull
pull remote changes (collaborators' changes to local repo)
```
git pull
```

#### 4. Common workflow of committing to remote branch.
Create a new branch.
```
git branch checkout -b <your branch name>
```
Work on the changes.
Pull all the remote changes and solve conflicts locally.
```
git pull
```
Add changed files. Repeat this until add all the files.
```
git add <file name>
```
Commit changes
```
git commit -m "some commit message about the change"
```
Push the changes to new branch
```
git push -u origin <a branch that not in the remote>
```
or push to an existing branch.
```
git push origin <a branch that not in the remote>
```
