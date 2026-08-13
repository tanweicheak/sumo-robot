# 1. Check which files were modified
git status

# 2. Stage all changed files
git add .

# 3. Commit changes with a message describing what you did
git commit -m "Added motor driver logic for sumo robot"

# 4. Push changes to GitHub
git push




Experimenting Safely (Branching)
# 1. Create and switch to a new experimental branch
git checkout -b feature/sharp-ir-sensor

# 2. Work, commit, and push on this branch
git add .
git commit -m "Added analog IR distance processing"
git push -u origin feature/sharp-ir-sensor

# 3. Switch back to your working main code anytime
git checkout main

# 4. Once the feature works completely, merge it into main
git checkout main
git merge feature/sharp-ir-sensor
git push


Inspecting and undoing mistakes 
# See exact line-by-line changes made since last commit
git diff

# View your recent commit history (press 'q' to exit)
git log --oneline -n 10

# Discard local changes in a specific file and revert to last commit
git restore src/motor_control.cpp

# Undo your last commit, but keep all modified files in your directory
git reset --soft HEAD~1


Version tagging
# 1. Create a tag for the current commit
git tag -a v1.0-competition -m "Sumo robot ready for competition round 1"

# 2. Push the tag to GitHub
git push origin v1.0-competition