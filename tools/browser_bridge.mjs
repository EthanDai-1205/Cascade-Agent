// browser_bridge.mjs: a persistent Playwright session that speaks JSON lines.
//
// Why a bridge instead of Playwright-for-Python: this project has no third-party Python
// dependencies and that is worth keeping. Node and Playwright are already present on this
// machine, so the bridge keeps the decision loop in stdlib Python and puts the browser
// behind a process boundary. One process, one browser, held open across steps, because
// re-launching per step would lose form state and page context.
//
// Protocol, one JSON object per line, in on stdin and out on stdout:
//   {"cmd":"state"}                     -> {"ok":true,"state":{...}}
//   {"cmd":"act","action":{"kind":...}}  -> {"ok":true,"detail":"..."}
//   {"cmd":"close"}                      -> {"ok":true} then exit
//
// Action kinds: click (by index), type, type_into (by index, with text), scroll_down,
// scroll_up, press_enter, press_escape, back, wait, goto, done. Everything the loop can
// ask for is in that list, and nothing else: the decision engine chooses, this file
// executes.
//
// The Playwright module is resolved from JEV_BROWSER_PLAYWRIGHT if set, so the bridge can
// point at a different install. It defaults to a bare "playwright", which Node resolves
// from the nearest node_modules above this file.

import { createInterface } from 'node:readline';

const SPEC = process.env.JEV_BROWSER_PLAYWRIGHT || 'playwright';
const MAX_LINES = Number(process.env.JEV_BROWSER_MAX_LINES || 18);
const MAX_CONTROLS = Number(process.env.JEV_BROWSER_MAX_CONTROLS || 40);

let chromium;
try {
  ({ chromium } = await import(SPEC));
} catch (err) {
  console.log(JSON.stringify({
    ok: false,
    error: `could not load Playwright from ${JSON.stringify(SPEC)}: ${err.message}. ` +
           'Install it (npm i playwright) or point JEV_BROWSER_PLAYWRIGHT at an install.',
  }));
  process.exit(3);
}

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : fallback;
};
const initialUrl = opt('url', 'about:blank');
const headed = args.includes('--headed');

const browser = await chromium.launch({ headless: !headed });
const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await context.newPage();
await page.goto(initialUrl, { waitUntil: 'domcontentloaded', timeout: 45000 }).catch(() => {});

async function readState() {
  await page.waitForTimeout(120);
  // One pass: the text an OCR pass would see, plus the controls the loop may click, each
  // tagged with a data-jc-idx hook so a later click does not depend on array order
  // surviving a re-render. Off-screen controls are marked rather than dropped, because
  // some real controls (a "More" link under a long list) are only reachable that way.
  const raw = await page.evaluate((limits) => {
    // Input types that can carry typed text. Anything else (checkbox, submit, range, ...)
    // is pressable, not fillable, and is never offered a typing option. This runs inside
    // the page, so the helper lives here rather than in Node scope.
    const TEXT_TYPES = new Set(['', 'text', 'search', 'email', 'url', 'tel', 'password', 'number', 'date']);
    const isTextField = (el) => {
      const tag = el.tagName.toLowerCase();
      if (tag === 'textarea' || el.isContentEditable) return true;
      return tag === 'input' && TEXT_TYPES.has((el.getAttribute('type') || '').toLowerCase());
    };
    const visible = (el) => {
      const r = el.getBoundingClientRect();
      const s = getComputedStyle(el);
      if (r.width < 4 || r.height < 4) return false;
      return s.visibility !== 'hidden' && s.display !== 'none' && parseFloat(s.opacity || '1') > 0.05;
    };

    document.querySelectorAll('[data-jc-idx]').forEach((el) => el.removeAttribute('data-jc-idx'));
    const controls = [];
    for (const el of document.querySelectorAll('a,button,input,select,textarea,[role=button],[role=link],[contenteditable=true]')) {
      if (!visible(el)) continue;
      const tag = el.tagName.toLowerCase();
      const role = el.getAttribute('role') || (tag === 'a' ? 'link' : tag === 'input' ? (el.type || 'input') : tag);
      const label = (el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.innerText ||
                     el.value || el.getAttribute('title') || el.name || '').trim().replace(/\s+/g, ' ').slice(0, 60);
      if (!label) continue;
      const key = `${role}|${label}`;
      if (controls.some((c) => c.key === key)) continue;
      const r = el.getBoundingClientRect();
      const onscreen = r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
      const i = controls.length + 1;
      el.setAttribute('data-jc-idx', String(i));
      controls.push({ i, role, label, onscreen, field: isTextField(el), key });
      if (controls.length >= limits.maxControls) break;
    }

    const lines = [];
    for (const el of document.querySelectorAll('h1,h2,h3,p,li,td,label,span,div')) {
      if (!visible(el)) continue;
      if (el.querySelector('h1,h2,h3,p,li,td,label')) continue;
      const text = (el.innerText || '').trim().replace(/\s+/g, ' ');
      if (text.length < 2 || text.length > 160) continue;
      lines.push(text);
      if (lines.length >= limits.maxLines) break;
    }

    const active = document.activeElement;
    let focused = 'none';
    if (active && active !== document.body) {
      const tag = active.tagName.toLowerCase();
      const label = active.getAttribute('aria-label') || active.getAttribute('placeholder') ||
                    active.name || tag;
      // The focused field's current value rides along, so a successful type is visible
      // as a state change: without it, a loop that fills a field reads as a no-op and
      // the two-no-op guard ends the run before it can press enter.
      const typeable = ['input', 'textarea'].includes(tag) || active.isContentEditable;
      const value = typeable
        ? String(active.value ?? active.innerText ?? '').trim().replace(/\s+/g, ' ').slice(0, 40)
        : '';
      focused = value ? `${label} = '${value}'` : label;
    }

    return { lines, controls, focused };
  }, { maxLines: MAX_LINES, maxControls: MAX_CONTROLS });

  return {
    url: page.url(),
    title: await page.title(),
    lines: raw.lines,
    controls: raw.controls.map(({ i, role, label, onscreen, field }) => ({ i, role, label, onscreen, field })),
    focused: raw.focused,
  };
}

async function perform(action) {
  const kind = String(action?.kind || '');
  if (kind === 'click') {
    const sel = `[data-jc-idx="${Number(action.index)}"]`;
    const el = await page.$(sel);
    if (!el) return { ok: false, error: `no control numbered ${action.index} on the page now` };
    const label = await el.evaluate((n) => (n.innerText || n.getAttribute('aria-label') || '').trim().slice(0, 60));
    await el.click({ timeout: 8000 }).catch(async () => { await el.click({ force: true, timeout: 8000 }); });
    await page.waitForTimeout(250);
    return { ok: true, detail: `clicked ${JSON.stringify(label)}` };
  }
  if (kind === 'type') {
    const text = String(action.text ?? '');
    if (!text) return { ok: false, error: 'no text to type' };
    const active = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return null;
      const tag = el.tagName.toLowerCase();
      if (!['input', 'textarea'].includes(tag) && !el.isContentEditable) return null;
      el.setAttribute('data-jc-focus', '1');
      return true;
    });
    if (!active) return { ok: false, error: 'nothing focusable is focused; click a field first' };
    await page.fill('[data-jc-focus="1"]', text);
    await page.evaluate(() => document.activeElement?.removeAttribute('data-jc-focus'));
    await page.waitForTimeout(150);
    return { ok: true, detail: `typed ${text.length} characters` };
  }
  if (kind === 'type_into') {
    const text = String(action.text ?? '');
    if (!text) return { ok: false, error: 'no text to type' };
    const sel = `[data-jc-idx="${Number(action.index)}"]`;
    const el = await page.$(sel);
    if (!el) return { ok: false, error: `no control numbered ${action.index} on the page now` };
    const fillable = await el.evaluate((n) => {
      const tag = n.tagName.toLowerCase();
      return ['input', 'textarea'].includes(tag) || n.isContentEditable;
    });
    if (!fillable) return { ok: false, error: 'that control cannot carry text' };
    await el.click({ timeout: 8000 }).catch(async () => { await el.click({ force: true, timeout: 8000 }); });
    await el.fill(text, { timeout: 8000 });
    await page.waitForTimeout(150);
    return { ok: true, detail: `typed ${text.length} characters into control ${action.index}` };
  }
  if (kind === 'back') {
    const response = await page.goBack({ waitUntil: 'domcontentloaded', timeout: 15000 }).catch(() => null);
    if (!response && page.url() === 'about:blank') return { ok: false, error: 'no page to go back to' };
    await page.waitForTimeout(200);
    return { ok: true, detail: `went back to ${page.url()}` };
  }
  if (kind === 'scroll_down' || kind === 'scroll_up') {
    const by = kind === 'scroll_down' ? 1 : -1;
    await page.evaluate((d) => window.scrollBy(0, d * Math.round(window.innerHeight * 0.8)), by);
    await page.waitForTimeout(200);
    return { ok: true, detail: kind.replace('_', ' ') };
  }
  if (kind === 'press_enter' || kind === 'press_escape') {
    const key = kind === 'press_enter' ? 'Enter' : 'Escape';
    await page.keyboard.press(key);
    await page.waitForTimeout(200);
    return { ok: true, detail: `pressed ${key}` };
  }
  if (kind === 'goto') {
    const url = String(action.url ?? '');
    if (!url) return { ok: false, error: 'no url to open' };
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
    return { ok: true, detail: `opened ${url}` };
  }
  if (kind === 'wait') {
    await page.waitForTimeout(700);
    return { ok: true, detail: 'waited' };
  }
  if (kind === 'done') return { ok: true, detail: 'done' };
  return { ok: false, error: `unsupported action ${JSON.stringify(kind)}` };
}

function reply(payload) {
  process.stdout.write(JSON.stringify(payload) + '\n');
}

const rl = createInterface({ input: process.stdin });
rl.on('line', async (line) => {
  const text = line.trim();
  if (!text) return;
  let message;
  try {
    message = JSON.parse(text);
  } catch (err) {
    reply({ ok: false, error: `could not parse the command as JSON: ${err.message}` });
    return;
  }
  try {
    if (message.cmd === 'state') reply({ ok: true, state: await readState() });
    else if (message.cmd === 'act') reply({ ok: true, ...(await perform(message.action)) });
    else if (message.cmd === 'close') {
      await browser.close();
      reply({ ok: true });
      process.exit(0);
    } else reply({ ok: false, error: `unsupported command ${JSON.stringify(message.cmd)}` });
  } catch (err) {
    reply({ ok: false, error: `${err.name}: ${String(err.message).slice(0, 200)}` });
  }
});
rl.on('close', async () => {
  await browser.close().catch(() => {});
  process.exit(0);
});
