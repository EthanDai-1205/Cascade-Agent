// macos_bridge.swift: the desktop counterpart of browser_bridge.mjs.
//
// The same trick as the browser, pointed at the screen instead of a page: the app
// supplies its own controls, with roles and labels, through the macOS Accessibility
// tree. No vision is involved, nothing is guessed from pixels, and the package that
// drives this stays dependency-free because the hard part lives here, behind the same
// one-JSON-object-per-line protocol.
//
// Protocol, one JSON object per line, in on stdin and out on stdout:
//   {"cmd":"state"}  -> {"ok":true,"state":{app, bundle, window, lines, controls, apps, focused}}
//   {"cmd":"act","action":{"kind":...}} -> {"ok":true,"detail":"..."} or {"ok":false,"error":"..."}
//   {"cmd":"check"}  -> {"ok":true,"trusted":bool,"hint":"..."}   (asks macOS to show the dialog)
//   {"cmd":"close"}  -> {"ok":true} then exit
//
// Action kinds: click (by index), set_value (by index, with text), type, key, scroll,
// switch (to a running app), launch, wait, done — plus press_enter / press_escape as
// sugar for key. Everything the loop can ask for is in that list, and nothing else:
// the decision engine chooses, this file executes.
//
// Controls are re-numbered on every state read, and an act re-walks the tree and
// resolves the index against the fresh walk — the same trick as the browser bridge's
// data-jc-idx, because a stale reference is how clicks land on controls that moved.
//
// Permission: reading another app's Accessibility tree needs the Accessibility grant,
// which macOS attaches to the process that runs this file (usually your terminal app).
// Without it the bridge exits with code 4 and a JSON error line saying so. No Screen
// Recording is requested, ever: this bridge never looks at pixels.

import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

let env = ProcessInfo.processInfo.environment
let MAX_CONTROLS = Int(env["JEV_COMPUTER_MAX_CONTROLS"] ?? "") ?? 25
let MAX_ELEMENTS = Int(env["JEV_COMPUTER_MAX_ELEMENTS"] ?? "") ?? 120
let MAX_LINES = Int(env["JEV_COMPUTER_MAX_LINES"] ?? "") ?? 18
let MAX_DEPTH = Int(env["JEV_COMPUTER_MAX_DEPTH"] ?? "") ?? 8

// Roles that respond to AXPress. Static text is excluded on purpose: its content
// already reaches the engine through the lines list, and offering every label as a
// clickable control floods the option set with near-duplicates.
let PRESSABLE: Set<String> = [
    "button", "checkbox", "radio button", "switch", "menu item", "menu bar item",
    "menu button", "pop up button", "link", "tab", "row", "incrementor",
]
let FIELDS: Set<String> = ["text field", "search field", "text area", "combo box"]

// Roles that are containers: never controls themselves, but walked through.
let CONTAINERS: Set<String> = [
    "application", "window", "sheet", "group", "scroll area", "split group", "splitter",
    "toolbar", "tab group", "table", "outline", "list", "layout area", "layout item",
    "menu bar", "menu", "cell", "generic element", "unknown", "ignored",
]

func friendlyRole(_ raw: String) -> String {
    let body = raw.hasPrefix("AX") ? String(raw.dropFirst(2)) : raw
    switch body {
    case "Button": return "button"
    case "CheckBox": return "checkbox"
    case "RadioButton": return "radio button"
    case "Switch": return "switch"
    case "TextField": return "text field"
    case "SearchField": return "search field"
    case "TextArea": return "text area"
    case "ComboBox": return "combo box"
    case "StaticText": return "static text"
    case "MenuItem": return "menu item"
    case "MenuBarItem": return "menu bar item"
    case "MenuBar": return "menu bar"
    case "MenuButton": return "menu button"
    case "PopUpButton": return "pop up button"
    case "PopUp": return "pop up"
    case "Link": return "link"
    case "Image": return "image"
    case "Slider": return "slider"
    case "TabGroup": return "tab group"
    case "Tab": return "tab"
    case "Row": return "row"
    case "Incrementor": return "incrementor"
    case "ScrollArea": return "scroll area"
    case "SplitGroup": return "split group"
    case "Splitter": return "splitter"
    case "Toolbar": return "toolbar"
    case "Sheet": return "sheet"
    case "Group": return "group"
    case "Application": return "application"
    case "Window": return "window"
    case "Table": return "table"
    case "Outline": return "outline"
    case "List": return "list"
    case "Cell": return "cell"
    case "Heading": return "heading"
    case "ProgressIndicator": return "progress indicator"
    case "ValueIndicator": return "value indicator"
    case "LayoutArea": return "layout area"
    case "LayoutItem": return "layout item"
    case "GenericElement": return "generic element"
    case "Unknown": return "unknown"
    case "HelpTag": return "help tag"
    case "GrowArea": return "grow area"
    case "Drawer": return "drawer"
    case "Ruler": return "ruler"
    case "ColorWell": return "color well"
    case "TimeField": return "text field"
    case "DatePicker": return "date picker"
    case "Stepper": return "incrementor"
    case "RelevanceIndicator": return "value indicator"
    case "RatingIndicator": return "value indicator"
    case "ContentGroup": return "group"
    case "Map": return "map"
    case "Browser": return "browser"
    default:
        // Camel case to words, so an unmapped role still reads sensibly.
        var words: [String] = []
        var current = ""
        for ch in body {
            if ch.isUppercase, !current.isEmpty {
                words.append(current.lowercased())
                current = String(ch)
            } else {
                current.append(ch)
            }
        }
        if !current.isEmpty { words.append(current.lowercased()) }
        return words.joined(separator: " ")
    }
}

func axAttr(_ el: AXUIElement, _ attr: String) -> CFTypeRef? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(el, attr as CFString, &value) == .success else { return nil }
    return value
}

func axString(_ el: AXUIElement, _ attr: String) -> String? {
    axAttr(el, attr) as? String
}

func axChildren(_ el: AXUIElement) -> [AXUIElement] {
    axAttr(el, kAXChildrenAttribute as String) as? [AXUIElement] ?? []
}

func axElement(_ el: AXUIElement, _ attr: String) -> AXUIElement? {
    guard let raw = axAttr(el, attr), CFGetTypeID(raw) == AXUIElementGetTypeID() else { return nil }
    return (raw as! AXUIElement)
}

func axPoint(_ el: AXUIElement, _ attr: String) -> CGPoint? {
    guard let raw = axAttr(el, attr), CFGetTypeID(raw) == AXValueGetTypeID() else { return nil }
    var point = CGPoint.zero
    guard AXValueGetValue(raw as! AXValue, .cgPoint, &point) else { return nil }
    return point
}

func axSize(_ el: AXUIElement, _ attr: String) -> CGSize? {
    guard let raw = axAttr(el, attr), CFGetTypeID(raw) == AXValueGetTypeID() else { return nil }
    var size = CGSize.zero
    guard AXValueGetValue(raw as! AXValue, .cgSize, &size) else { return nil }
    return size
}

// MARK: state collection

final class Collector {
    var controls: [[String: Any]] = []
    var elements: [AXUIElement] = []  // index-aligned with controls; 1-based outside
    var lines: [String] = []
    private var seenLines = Set<String>()
    let screenBounds = CGDisplayBounds(CGMainDisplayID())

    var full: Bool { controls.count >= MAX_CONTROLS || elements.count >= MAX_ELEMENTS }

    func addLine(_ raw: String) {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.count >= 2, text.count <= 160, !seenLines.contains(text), lines.count < MAX_LINES
        else { return }
        seenLines.insert(text)
        lines.append(text)
    }
}

func controlLabel(_ el: AXUIElement, role: String) -> String {
    var candidates = [kAXTitleAttribute, kAXDescriptionAttribute, kAXHelpAttribute]
    if role == "text field" || role == "search field" || role == "text area" || role == "combo box" {
        candidates.insert(kAXValueAttribute, at: 0)  // many fields are labelled by their value
    }
    for attr in candidates {
        if let text = axString(el, attr as String) {
            let cleaned = text.split(separator: "\t").first.map(String.init) ?? text  // "New Window\t⌘N" -> "New Window"
            let trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                return String(trimmed.prefix(60))
            }
        }
    }
    return ""
}

func walk(_ el: AXUIElement, depth: Int, menuPath: String, _ collector: Collector) {
    guard depth <= MAX_DEPTH, !collector.full else { return }
    let rawRole = axString(el, kAXRoleAttribute as String) ?? ""
    let role = friendlyRole(rawRole)

    if rawRole == "AXStaticText" || rawRole == "AXHeading" {
        let text = axString(el, kAXValueAttribute as String)
            ?? axString(el, kAXTitleAttribute as String) ?? ""
        collector.addLine(text)
    }

    if role != "static text" && role != "heading" && !CONTAINERS.contains(role) {
        let pressable = PRESSABLE.contains(role)
        let field = FIELDS.contains(role)
        if pressable || field {
            let label = controlLabel(el, role: role)
            if !label.isEmpty {
                var onscreen = true
                if menuPath.isEmpty {
                    // Menu items have no meaningful frame until their menu is open, so
                    // the on-screen check only applies to controls in a window. A closed
                    // menu's items are perfectly pressable via AXPress.
                    if let p = axPoint(el, kAXPositionAttribute as String),
                       let s = axSize(el, kAXSizeAttribute as String) {
                        let frame = CGRect(origin: p, size: s)
                        onscreen = frame.intersects(collector.screenBounds) && s.width >= 2 && s.height >= 2
                    }
                }
                var control: [String: Any] = [
                    "i": collector.controls.count + 1,
                    "role": role,
                    "label": menuPath.isEmpty ? label : "\(menuPath) > \(label)",
                    "onscreen": onscreen,
                    "field": field,
                ]
                if field, let value = axString(el, kAXValueAttribute as String), !value.isEmpty {
                    control["value"] = String(value.prefix(60))
                }
                collector.controls.append(control)
                collector.elements.append(el)
            }
        }
    }

    for child in axChildren(el) {
        walk(child, depth: depth + 1, menuPath: menuPath, collector)
        if collector.full { return }
    }
}

// kAXMenuAttribute is one of the Accessibility constants Swift never imported;
// the string value is stable and public, so it is named here.
let AXMenuAttribute = "AXMenu"

// The menu bar is walked specially: menu bar items first, then one level of items
// under each, so "File > New Window" becomes one control rather than two questions.
// A bar item with exposed children is not offered as a click itself: next to
// "File > New Note" it is the near-duplicate that splits the engine's probability,
// and AXPress on the leaf item performs the action without the menu being open.
// The menu element hides one level down under AXChildren on a closed item — the
// AXMenu attribute only exists in some states — so both paths are tried.
func walkMenuBar(_ app: AXUIElement, _ collector: Collector) {
    guard let bar = axElement(app, kAXMenuBarAttribute as String) else { return }
    // The Apple menu and the app's own menu (About, Settings, Services, Quit) are
    // system baggage: together they flood the control budget before the app's real
    // File and Edit menus are reached. A walk starts with the app's own menus.
    let appName = axString(app, kAXTitleAttribute as String) ?? ""
    for item in axChildren(bar) {
        guard !collector.full else { return }
        let title = (axString(item, kAXTitleAttribute as String) ?? "")
            .split(separator: "\t").first.map(String.init) ?? ""
        guard !title.isEmpty else { continue }
        guard title != "Apple", title != appName else { continue }
        let menu = axElement(item, AXMenuAttribute) ?? axChildren(item).first
        let children = menu.map { axChildren($0) } ?? []
        if children.isEmpty && collector.controls.count < MAX_CONTROLS {
            collector.controls.append([
                "i": collector.controls.count + 1, "role": "menu bar item",
                "label": title, "onscreen": true, "field": false,
            ])
            collector.elements.append(item)
        }
        // Eight per menu: enough for any single menu's common items, few enough that
        // the app menu cannot crowd File and Edit out of the control budget.
        for child in children.prefix(8) {
            guard !collector.full else { return }
            walk(child, depth: 1, menuPath: title, collector)
        }
    }
}

struct DesktopState {
    let appName: String
    let bundle: String
    let window: String
    let lines: [String]
    let controls: [[String: Any]]
    let apps: [String]
    let focused: String
}

func readState() -> [String: Any] {
    guard let app = NSWorkspace.shared.frontmostApplication else {
        return [
            "app": "unknown", "bundle": "", "window": "", "lines": [],
            "controls": [[String: Any]](), "apps": [], "focused": "none",
        ]
    }
    let appName = app.localizedName ?? "unknown"
    let bundle = app.bundleIdentifier ?? ""
    let element = AXUIElementCreateApplication(app.processIdentifier)

    let windowTitle = { () -> String in
        let focused = axElement(element, kAXFocusedWindowAttribute as String)
            ?? axElement(element, kAXMainWindowAttribute as String)
        guard let window = focused else { return "" }
        return axString(window, kAXTitleAttribute as String) ?? ""
    }()

    let collector = Collector()
    walkMenuBar(element, collector)
    if let window = axElement(element, kAXFocusedWindowAttribute as String)
        ?? axElement(element, kAXMainWindowAttribute as String) {
        walk(window, depth: 0, menuPath: "", collector)
    }

    let focusedText = { () -> String in
        guard let focused = axElement(element, kAXFocusedUIElementAttribute as String) else { return "none" }
        let role = friendlyRole(axString(focused, kAXRoleAttribute as String) ?? "")
        var label = controlLabel(focused, role: role)
        if FIELDS.contains(role) {
            // The tail, not the head: typing appends, so the recent end is the end that
            // moves, and a state whose focused field shows the same head every step
            // reads as a no-op while the text piles up.
            let full = (axString(focused, kAXValueAttribute as String) ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !full.isEmpty {
                label = full.count > 40 ? "…" + String(full.suffix(40)) : full
            }
        }
        return label.isEmpty ? role : "\(role) '\(label)'"
    }()

    let running = NSWorkspace.shared.runningApplications
        .filter { $0.activationPolicy == .regular && $0.localizedName != nil }
        .prefix(15)
        .map { $0.localizedName! }

    return [
        "app": appName,
        "bundle": bundle,
        "window": windowTitle,
        "lines": collector.lines,
        "controls": collector.controls,
        "apps": Array(running),
        "focused": focusedText,
    ]
}

// MARK: acting

func resolveControl(_ index: Int) -> (AXUIElement, [String: Any])? {
    guard index >= 1 else { return nil }
    let collector = Collector()
    if let app = NSWorkspace.shared.frontmostApplication {
        let element = AXUIElementCreateApplication(app.processIdentifier)
        walkMenuBar(element, collector)
        if let window = axElement(element, kAXFocusedWindowAttribute as String)
            ?? axElement(element, kAXMainWindowAttribute as String) {
            walk(window, depth: 0, menuPath: "", collector)
        }
    }
    guard index <= collector.elements.count else { return nil }
    return (collector.elements[index - 1], collector.controls[index - 1])
}

func elementCenter(_ el: AXUIElement) -> CGPoint? {
    guard let p = axPoint(el, kAXPositionAttribute as String),
          let s = axSize(el, kAXSizeAttribute as String) else { return nil }
    return CGPoint(x: p.x + s.width / 2, y: p.y + s.height / 2)
}

func mouseClick(at point: CGPoint) {
    let source = CGEventSource(stateID: .hidSystemState)
    CGEvent(mouseEventSource: source, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left)?
        .post(tap: .cghidEventTap)
    usleep(40_000)
    for kind in [CGEventType.leftMouseDown, CGEventType.leftMouseUp] {
        CGEvent(mouseEventSource: source, mouseType: kind, mouseCursorPosition: point, mouseButton: .left)?
            .post(tap: .cghidEventTap)
        usleep(40_000)
    }
}

func postKey(_ keyCode: CGKeyCode, flags: CGEventFlags) {
    let source = CGEventSource(stateID: .hidSystemState)
    let down = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true)
    let up = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false)
    down?.flags = flags
    up?.flags = flags
    down?.post(tap: .cghidEventTap)
    usleep(10_000)
    up?.post(tap: .cghidEventTap)
    usleep(10_000)
}

func typeString(_ text: String) {
    let source = CGEventSource(stateID: .hidSystemState)
    for ch in text {
        if ch == "\n" || ch == "\r" {
            postKey(36, flags: [])
            continue
        }
        if ch == "\t" {
            postKey(48, flags: [])
            continue
        }
        let utf16 = Array(String(ch).utf16)
        let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true)
        let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false)
        down?.keyboardSetUnicodeString(stringLength: utf16.count, unicodeString: utf16)
        up?.keyboardSetUnicodeString(stringLength: utf16.count, unicodeString: utf16)
        down?.post(tap: .cghidEventTap)
        usleep(6_000)
        up?.post(tap: .cghidEventTap)
        usleep(6_000)
    }
}

// Standard US keycodes, because a key name has to become a virtual key somehow.
let NAMED_KEYS: [String: CGKeyCode] = [
    "return": 36, "enter": 36, "escape": 53, "esc": 53, "tab": 48, "delete": 51,
    "backspace": 51, "forward delete": 117, "space": 49, "up": 126, "down": 125,
    "left": 123, "right": 124, "home": 115, "end": 119, "page up": 116, "page down": 121,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98,
    "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 105,
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "1": 18, "2": 19, "3": 20, "4": 21, "5": 23, "6": 22, "7": 26, "8": 28, "9": 25, "0": 29,
    "-": 27, "=": 24, "[": 33, "]": 30, "\\": 42, ";": 41, "'": 39, ",": 43, ".": 47, "/": 44, "`": 50,
]
let MODIFIERS: [String: CGEventFlags] = [
    "cmd": .maskCommand, "command": .maskCommand, "ctrl": .maskControl, "control": .maskControl,
    "opt": .maskAlternate, "alt": .maskAlternate, "option": .maskAlternate,
    "shift": .maskShift, "fn": .maskSecondaryFn,
]

func parseKey(_ name: String) -> (CGKeyCode, CGEventFlags)? {
    var flags: CGEventFlags = []
    var key = name.lowercased().trimmingCharacters(in: .whitespaces)
    while let sep = key.firstIndex(of: "+") {
        let mod = String(key[..<sep]).trimmingCharacters(in: .whitespaces)
        guard let flag = MODIFIERS[mod] else { return nil }
        flags.insert(flag)
        key = String(key[key.index(after: sep)...]).trimmingCharacters(in: .whitespaces)
    }
    guard let code = NAMED_KEYS[key] else { return nil }
    return (code, flags)
}

func scroll(_ direction: String) -> String {
    let lines: Int32 = direction == "up" ? 5 : -5
    CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1,
            wheel1: lines, wheel2: 0, wheel3: 0)?.post(tap: .cghidEventTap)
    usleep(150_000)
    return "scrolled \(direction)"
}

func switchTo(_ name: String) -> String? {
    let wanted = name.lowercased()
    let hit = NSWorkspace.shared.runningApplications.first {
        ($0.localizedName ?? "").lowercased() == wanted
    } ?? NSWorkspace.shared.runningApplications.first {
        ($0.localizedName ?? "").lowercased().hasPrefix(wanted)
    }
    guard let app = hit else { return nil }
    return app.activate(options: []) ? "switched to \(app.localizedName ?? name)" : nil
}

func launchApp(_ name: String) -> String? {
    if NSWorkspace.shared.launchApplication(withBundleIdentifier: name, options: [],
                                            additionalEventParamDescriptor: nil,
                                            launchIdentifier: nil) {
        return "launched \(name) by bundle id"
    }
    if NSWorkspace.shared.launchApplication(name) {
        return "launched \(name); it may still be opening"
    }
    return nil
}

func perform(_ action: [String: Any]) -> [String: Any] {
    let kind = (action["kind"] as? String) ?? ""
    func fail(_ message: String) -> [String: Any] {
        ["ok": false, "error": message]
    }

    if kind == "wait" {
        usleep(700_000)
        return ["ok": true, "detail": "waited"]
    }
    if kind == "done" {
        return ["ok": true, "detail": "done"]
    }
    if kind == "scroll" {
        let direction = (action["direction"] as? String) ?? "down"
        guard direction == "up" || direction == "down" else {
            return fail("scroll direction must be up or down")
        }
        return ["ok": true, "detail": scroll(direction)]
    }
    if kind == "press_enter" || kind == "press_escape" {
        let name = kind == "press_enter" ? "return" : "escape"
        guard let (code, flags) = parseKey(name) else { return fail("unknown key \(name)") }
        postKey(code, flags: flags)
        return ["ok": true, "detail": "pressed \(name)"]
    }
    if kind == "key" {
        let name = (action["name"] as? String) ?? ""
        guard let (code, flags) = parseKey(name) else {
            return fail("unknown key \(name); use a named key (enter, escape, tab, ...) or a combo like cmd+w")
        }
        postKey(code, flags: flags)
        return ["ok": true, "detail": "pressed \(name)"]
    }
    if kind == "type" {
        let text = (action["text"] as? String) ?? ""
        guard !text.isEmpty else { return fail("no text to type") }
        // Keystrokes go to the key window whatever it is, and a spray into a list, a
        // table, or a menu is how real damage happens. Type only when a text-carrying
        // element is confirmed focused; otherwise refuse, like the browser bridge does.
        let textable = ["text field", "text area", "search field", "combo box"]
        if let app = NSWorkspace.shared.frontmostApplication {
            let ax = AXUIElementCreateApplication(app.processIdentifier)
            if let focused = axElement(ax, kAXFocusedUIElementAttribute as String) {
                let role = friendlyRole(axString(focused, kAXRoleAttribute as String) ?? "")
                guard textable.contains(role) else {
                    return fail("nothing textable is focused (focused role is \(role)); create or click a field first")
                }
            } else {
                return fail("nothing is focused; create or click a field first")
            }
        }
        typeString(text)
        return ["ok": true, "detail": "typed \(text.count) characters"]
    }
    if kind == "switch" {
        let name = (action["app"] as? String) ?? ""
        guard let detail = switchTo(name) else {
            return fail("no running app called \(name)")
        }
        usleep(400_000)  // give the activation a beat before the next state read
        return ["ok": true, "detail": detail]
    }
    if kind == "launch" {
        let name = (action["app"] as? String) ?? ""
        guard let detail = launchApp(name) else { return fail("could not launch \(name)") }
        usleep(1_200_000)  // a fresh app needs a moment before its tree is worth reading
        return ["ok": true, "detail": detail]
    }

    // Everything below needs a control resolved by index against a fresh walk.
    let index = (action["index"] as? NSNumber)?.intValue ?? 0
    guard let (element, control) = resolveControl(index) else {
        return fail("no control numbered \(index) on screen now")
    }
    let label = (control["label"] as? String) ?? ""

    if kind == "click" {
        let pressError = AXUIElementPerformAction(element, kAXPressAction as CFString)
        if pressError == .success {
            usleep(200_000)
            return ["ok": true, "detail": "clicked \(label)"]
        }
        // Not every pressable control answers AXPress; a real mouse click at the
        // control's own position is the fallback, never a guessed coordinate.
        guard let center = elementCenter(element) else {
            return fail("could not press or locate \(label) (AXPress error \(pressError.rawValue))")
        }
        mouseClick(at: center)
        usleep(150_000)
        return ["ok": true, "detail": "clicked \(label) with a mouse event"]
    }
    if kind == "focus" {
        let error = AXUIElementSetAttributeValue(
            AXUIElementCreateApplication(NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0),
            kAXFocusedUIElementAttribute as CFString, element)
        if error == .success {
            return ["ok": true, "detail": "focused \(label)"]
        }
        guard let center = elementCenter(element) else {
            return fail("could not focus or locate \(label)")
        }
        mouseClick(at: center)
        return ["ok": true, "detail": "focused \(label) with a mouse event"]
    }
    if kind == "set_value" {
        let text = (action["text"] as? String) ?? ""
        guard !text.isEmpty else { return fail("no text to set") }
        let error = AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, text as CFTypeRef)
        if error == .success {
            return ["ok": true, "detail": "set \(label) to \(text.count) characters"]
        }
        // Some fields refuse a direct write; focus them and type instead.
        AXUIElementPerformAction(element, kAXRaiseAction as CFString)
        typeString(text)
        return ["ok": true, "detail": "typed \(text.count) characters into \(label)"]
    }
    return fail("unsupported action \(kind)")
}

// MARK: the protocol loop

func reply(_ payload: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: payload),
          let line = String(data: data, encoding: .utf8) else { return }
    FileHandle.standardOutput.write((line + "\n").data(using: .utf8)!)
}

func permissionHint() -> String {
    let name = NSWorkspace.shared.frontmostApplication?.localizedName ?? "your terminal"
    return "this bridge needs the Accessibility grant: open System Settings > Privacy & Security > "
        + "Accessibility and add the app that runs it (often \(name) or your terminal), then try again. "
        + "`computer --check-permissions` asks macOS to show the dialog."
}

// --app=NAME activates the named app before the first state read, so a run can
// begin somewhere other than whatever happened to be frontmost.
// --check-only skips the startup permission guard, so `check` can be the thing
// that asks macOS to show its dialog instead of the bridge dying before stdin.
for argument in CommandLine.arguments.dropFirst() where argument.hasPrefix("--app=") {
    let wanted = String(argument.dropFirst("--app=".count))
    if switchTo(wanted) != nil {
        // Activation is asynchronous and can lose the race with the first state read,
        // which makes the engine's first decision an unnecessary app switch. Poll
        // until the frontmost app really is the wanted one, re-asking on the way.
        for _ in 0..<10 {
            usleep(400_000)
            if NSWorkspace.shared.frontmostApplication?.localizedName == wanted { break }
            _ = switchTo(wanted)
        }
    }
}
let checkOnly = CommandLine.arguments.contains("--check-only")

guard checkOnly || AXIsProcessTrustedWithOptions(
    [kAXTrustedCheckOptionPrompt.takeRetainedValue() as String: false] as CFDictionary
) else {
    reply([
        "ok": false,
        "error": "the Accessibility permission is missing. " + permissionHint(),
    ])
    exit(4)
}

while let line = readLine() {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { continue }
    guard let data = trimmed.data(using: .utf8),
          let message = try? JSONSerialization.jsonObject(with: data),
          let object = message as? [String: Any] else {
        reply(["ok": false, "error": "could not parse the command as JSON"])
        continue
    }
    switch (object["cmd"] as? String) ?? "" {
    case "state":
        reply(["ok": true, "state": readState()])
    case "act":
        let outcome = perform((object["action"] as? [String: Any]) ?? [:])
        reply(outcome)
    case "check":
        // The one command that may show the system prompt; it exists so the CLI can
        // walk the user through the grant instead of failing cryptically.
        let trusted = AXIsProcessTrustedWithOptions(
            [kAXTrustedCheckOptionPrompt.takeRetainedValue() as String: true] as CFDictionary
        )
        reply(["ok": true, "trusted": trusted, "hint": trusted ? "" : permissionHint()])
    case "close":
        reply(["ok": true])
        exit(0)
    default:
        reply(["ok": false, "error": "unsupported command \((object["cmd"] as? String) ?? "")"])
    }
}
exit(0)
