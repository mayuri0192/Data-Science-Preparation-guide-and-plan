```git config --list```

Check if all details match. 

Use below commands for changing details. 

```bash 
git config --global user.email youremailhere
```

```bash 
git config --global user.name "yourusername"
```

or 

```nash 
git config user.email "email address"
```
```bash 
git config user.name "Name"
```

```bash 
git config user.password "password" 
```

The commands `git config user.email "email address"` and `git config --global user.email youremailhere` are both used to configure the email address associated with your Git commits, but they have different scopes and purposes:

1. **`git config user.email "email address"`**:
   This command sets the email address that will be used for your commits within the specific Git repository where you run the command. The email address will only apply to the current repository and will not affect other repositories on your system.

   For example:
   ```bash
   git config user.email "john@example.com"
   ```

2. **`git config --global user.email youremailhere`**:
   This command sets the global email address that will be used for commits across all Git repositories on your system. The `--global` flag ensures that the configuration change applies globally to all repositories unless overridden by a local configuration.

   For example:
   ```bash
   git config --global user.email "john@example.com"
   ```

In both cases, it's important to set your email address correctly to ensure proper attribution of your commits. The global configuration is useful when you consistently use the same email address across multiple repositories. However, within a specific repository, you might want to use different email addresses for different projects or collaborators. Setting the local configuration within a repository allows for that flexibility.

If you use the global configuration, remember that it applies to all repositories, so make sure you're comfortable with that level of consistency across your work.