# Editing the ChaBo website — no HTML needed

The entire website content lives in one file: `_config.yml`

Open it in any text editor (even Notepad works).
You'll see clearly labelled sections. Just change the text and save.

---

## The golden rule

Only edit text that appears **after a colon**, on lines that don't start with `#`.

```yaml
title: ChaBo           ← change "ChaBo" if you want
# This is a comment    ← lines starting with # are ignored, don't touch
```

---

## Changing the hero (top of page)

```yaml
hero:
  line1: "Your knowledge base."      ← change this text
  line2: "Your infrastructure."      ← change this text
  line3: "Your chatbot."             ← change this text (shows in red)
  subtitle: >
    ChaBo is an open-source...       ← change this paragraph
```

The `>` symbol just means "this is a long paragraph". Keep it there, just change the words after it.

---

## Changing links

```yaml
github_repo: "https://github.com/chabo-project/chabo"   ← your repo URL
readme_url:  "https://github.com/chabo-project/chabo/blob/main/README.md"
```

Just replace the URLs inside the quotes.

---

## Adding a new feature card

Find the `features:` section and copy-paste one block:

```yaml
    - icon:   "🔧"
      title:  "Your new feature title"
      desc:   "Your description here."
      jargon: "Technical tag here"
```

Make sure the spacing (indentation) matches the blocks above it exactly.

---

## Adding a new use case

Find `usecases:` and copy-paste:

```yaml
    - title: "Your use case title"
      desc:  "Your description here."
```

---

## Changing the badge pills in the hero

```yaml
  badges:
    - "Cloud agnostic"
    - "No vendor lock-in"
    - "Data sovereign"
    - "Self-hostable"       ← add, remove, or edit any of these
```

---

## After editing

1. Save `_config.yml`
2. Commit and push to GitHub
3. Wait ~30 seconds
4. Your site at `https://chabo-project.github.io/chabo/` updates automatically

---

## If something looks broken

- Check that your indentation (spaces) matches the lines around it
- Make sure quotes are in pairs: `"like this"` not `"like this`
- Never use Tab — only spaces for indentation

Still stuck? Just paste the broken section into Claude and ask what's wrong.
