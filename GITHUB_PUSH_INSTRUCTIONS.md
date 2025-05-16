# Instructions for Pushing to GitHub

Follow these steps to push the Hunyuan3D Glasses Generation project to GitHub:

## 1. Create a GitHub Repository

1. Go to https://github.com/amin8452
2. Click on "New" to create a new repository
3. Name it "Hunyuan3D-Glasses-Generation"
4. Add a description (optional)
5. Make it public
6. Do not initialize with README, .gitignore, or license (we already have these files)
7. Click "Create repository"

## 2. Initialize Git Repository Locally

Open a terminal/command prompt in the project directory and run:

```bash
git init
git add .
git commit -m "Initial commit: Hunyuan3D Glasses Generation"
```

## 3. Connect to GitHub and Push

```bash
git remote add origin https://github.com/amin8452/Hunyuan3D-Glasses-Generation.git
git branch -M main
git push -u origin main
```

## 4. Verify the Repository

1. Go to https://github.com/amin8452/Hunyuan3D-Glasses-Generation
2. Make sure all files are uploaded correctly
3. Check that the README is displayed properly

## 5. Add a Demo GIF

1. Create a demo GIF showing the tool in action
2. Place it in the `demo` folder as `demo.gif`
3. Push the changes:

```bash
git add demo/demo.gif
git commit -m "Add demo GIF"
git push
```

## 6. Share the Repository

Share the repository URL with users who want to use the tool:
https://github.com/amin8452/Hunyuan3D-Glasses-Generation
