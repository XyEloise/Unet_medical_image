# usage of git
click 'git bash here' at the root document,and setup a global deploy(for git to recognize the commit author):
git config --global user.name "name" 
git config --global user.email "youremail@a.com"

## set repositories
git init
git add .
git commit -m 'initial'
git remote add origin add_ssh_address(use 'git remote -v' check,ssh can avoid login procedure,if it's http address,use 'git remote rm origin')

!!!remember:if you need push/pull operation,change root to the current document

## modify repositories(everytime repeat again)
git add 'new_document'/git rm --cached 'new_document'
git commit -m 
git push origin [local beanch]:[remote branch](if error:src refsec master does not match any:repeat add . and commit opration then push)

## when local repositorie different from remote:
need to update local before push:'git pull --rebase origin master'

## fatal:refusing to merge unrelated histories:
'git pull origin master --allow-unrelated-histories'

## two ways to push local branch to remote branch
(1)setup remote branch then pull local:
git checkout -b new_remote_branch origin/new_remote_branch
(2)setup local branch then push remote
git checkout -b new_remote_branch (set and change to new branch)
git push origin local_new_remote_branch:remote_new_remote_branch (push local new branch to remote)

