# My_obsidian

A repo to save my obsidian notes without having to sync my email to it




---
# Global Progress

```dataviewjs
let tasks = dv.pages().file.tasks;
let total = tasks.length;
let done = tasks.where(t => t.completed).length;

if (total > 0) {
  let percent = Math.round((done / total) * 100);
  let filled = "█".repeat(Math.floor(percent / 10));
  let empty = "░".repeat(10 - Math.floor(percent / 10));

  dv.header(3, "Overall Progress");
  dv.paragraph(`${done}/${total} tasks completed`);
  dv.paragraph(`[${filled}${empty}] ${percent}%`);
} else {
  dv.paragraph("No tasks found.");
}
````

---

## Progress by Note

```dataviewjs
for (let page of dv.pages()) {
  let total = page.file.tasks.length;
  let done = page.file.tasks.where(t => t.completed).length;

  if (total > 0) {
    let percent = Math.round((done / total) * 100);

    // Progress bar format
    let barWidth = 30; // number of slots in the bar
    let filled = Math.round((percent / 100) * barWidth);
    let bar = "▮".repeat(filled) + "▯".repeat(barWidth - filled);

    // Display
    dv.header(4, `[${page.file.name}](${page.file.path})`);
    dv.paragraph(`*Path:* \`${page.file.path}\``);
    dv.paragraph(`${done}/${total} tasks completed`);
    dv.paragraph(`${bar} ${percent}%`);
    dv.paragraph("---");
  }
}

```

---


