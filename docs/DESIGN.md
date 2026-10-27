# Product design and usability rationale

The audience is a student, professor or technician reviewing battery complaints,
not a developer inspecting a service. The redesign borrows Apple's emphasis on
clear hierarchy, consistent navigation and readable data without copying Apple
branding, logos or proprietary assets.

## Task-first information architecture

Five persistent navigation destinations match user questions: an overview, a
complaint classifier, a degradation estimate, model results and searchable
records. There is no requirement to scroll past every chart to find a form.
The browser URL retains the selected view, back/forward navigation works, and
forms stay mounted so switching pages does not discard typed work. The overview
provides two direct task links. Exploratory clustering sits behind a disclosure
in Model results because it needs more explanation than the main tasks.

## Visual hierarchy and color

A soft gray canvas (#F5F6F8), white surfaces, graphite text (#1D2633) and a single
blue action color (#0067D6) create a restrained, Apple-inspired palette. Teal and
muted orange distinguish supporting measurements. There is no purple or
gradient. The system font stack uses the reader's native sans serif, with large
page titles, compact supporting text, tabular numbers and generous spacing.
Round corners and shallow shadows group related content without heavy borders.

Charts use a zero baseline, direct labels where practical, minimal grid lines,
short titles and nearby explanations. Horizontal complaint bars keep long labels
readable. The battery-health line uses the full 0-100% scale so small synthetic
differences cannot look like a dramatic decline. Text labels carry meaning as
well as color. Severity colors are muted and labels remain visible.

## Progressive explanation

The main forms ask for the user's task, not endpoint names or file paths. Help
panels explain what to enter and how to interpret the result. Probability bars
show uncertainty; review flags require a human check. The degradation form
explains the fast-charge share and percentage-point output. Model-results
terminology and validation detail live behind a plain-language disclosure.
Synthetic-data disclosure stays visible in the workspace and results.

## Access and responsiveness

Persistent navigation becomes a horizontally scrollable labelled navigation row
on narrow screens. Cards, forms and data tables adapt independently. There is
a skip link, visible keyboard focus, a focused heading after navigation, live
result announcements, labelled fields, explicit loading/error states, and a
retry action. Color alone never determines a status. Reduced-motion preference
removes transitions. This is an implementation rationale, not a claim of a
formal accessibility audit or user study.

## Screenshot and evaluation method

On an ultrawide display, capture the working application and crop to its bounded
content region. Exclude browser chrome, unrelated tabs and desktop whitespace.
Use separate captures of the overview, classification, regression and results;
never stretch the full ultrawide frame into a portrait report. Evaluate task
completion, not just visual appearance: select a destination, submit a valid
input, inspect its result, navigate away/back, and check that input persists.

Design references: Apple Human Interface Guidelines, Sidebars and Navigation
and search (accessed 27 October 2026):
https://developer.apple.com/design/human-interface-guidelines/sidebars
https://developer.apple.com/design/human-interface-guidelines/navigation-and-search
