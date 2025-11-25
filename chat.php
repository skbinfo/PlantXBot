<DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChatBot | Plant Regulatory Elements</title>
<link rel="icon" href="images/favicon/favicon.png" type="image/png">
<style>
/* ===== GLOBAL & TYPOGRAPHY ===== */
:root {
    --brand-blue: #004c97;
    --mid-blue:   #0063c6;
    --light-blue: #1490ff;
    --text-dark:  #333;
    --text-light: #fff;
    --gray:       #f7f8fa;
    --border-color: #e9ecef;
    --transition: 0.3s ease-in-out;
    --radius: 0.75rem;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --max-w: 1200px;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.7;
    color: var(--text-dark);
    background-color: var(--gray);
    margin: 0;
    padding: 0;
}

.container {
    max-width: var(--max-w);
    margin: 0 auto;
    padding: 0 1rem;
}

h1, h2, h3 {
    color: var(--brand-blue);
    line-height: 1.2;
    font-weight: 700;
}

h1 { font-size: 2.8rem; text-align: center; margin-bottom: 1rem; }
h2 { font-size: 2.2rem; text-align: center; margin-bottom: 2.5rem; border-bottom: 2px solid var(--border-color); padding-bottom: 1rem; }
h3 { font-size: 1.5rem; color: var(--mid-blue); margin-bottom: 1rem; }

p { margin-bottom: 1rem; max-width: 80ch; }
a { color: var(--mid-blue); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ===== NAVBAR ===== */
.navbar {
    background: var(--brand-blue);
    padding: 1rem;
    position: sticky;
    top: 0;
    z-index: 1000;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.navbar .container {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.brand-logo-title { display: flex; align-items: center; }
.logo-circle {
    width: 50px; height: 50px;
    border-radius: 50%; object-fit: cover;
    margin-right: 15px;
}
.brand {
    font-size: 1.5rem; font-weight: bold;
    color: var(--text-light); text-decoration: none;
}
nav a {
    margin-left: 20px; text-decoration: none;
    color: var(--text-light); font-size: 1rem; font-weight: 500;
    transition: var(--transition);
}
nav a:hover { opacity: 0.8; }

/* Sidebar specific styles */
.left-sidebar .db-card {
    display: block;
    margin-bottom: 1rem;
    text-decoration: none;
    color: inherit;
    border: 1px solid #e0e0e0;
    border-radius: var(--radius);
    overflow: hidden;
    transition: var(--transition);
}
.left-sidebar .db-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}
.left-sidebar .db-card img {
    width: 100%;
    height: auto;
    border-bottom: 1px solid #e0e0e0;
}
.left-sidebar .db-name {
    margin-top: 0.5rem;
    font-size: 1rem;
    color: var(--brand-blue);
    padding: 0.5rem;
    text-align: center;
}
.right-sidebar ul {
    list-style: none;
    padding: 0;
}
.right-sidebar li {
    margin-bottom: 1rem;
    padding: 0.75rem;
    background: #e6f0fa;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    transition: var(--transition);
}
.right-sidebar li:hover {
    background-color: #d0e0f5;
}
.right-sidebar a {
    color: var(--brand-blue);
    text-decoration: none;
}
.right-sidebar a:hover {
    text-decoration: underline;
}
.model-links {
    margin-top: 5rem;
}
.model-links a {
    display: block;
    margin-bottom: 0.5rem;
    color: var(--brand-blue);
    text-decoration: none;
}
.model-links a:hover {
    text-decoration: underline;
}

/* Main layout */
.main-container {
    display: flex;
    width: calc(100% - 4rem);
    max-width: 1800px;
    margin: 2rem auto;
    gap: 2rem;
}
.sidebar {
    flex: 0 0 250px;
    background: var(--text-light);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 1.5rem;
    position: sticky;
    top: 2rem;
}
.sidebar h3 {
    color: var(--brand-blue);
    margin-bottom: 1rem;
    font-size: 1.5rem;
    text-align: center;
}
.chat-center {
    flex: 1;
    min-width: 0;
}

/* Chat component styles */
.chat-wrapper {
    background: var(--text-light);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 2rem;
}
#chatbot-title {
    color: var(--brand-blue);
    text-align: center;
    margin-bottom: 0.25rem;
    font-size: clamp(2rem, 5vw, 2.5rem);
    font-weight: 700;
}
#chat-container {
    height: 400px;
    overflow-y: auto;
    scroll-behavior: smooth;
    border: 1px solid #e0e0e0;
    border-radius: var(--radius);
    padding: 1.5rem;
    background-color: #fdfdfd;
}

/* Bot introductory paragraph */
.bot-intro {
    color: #000;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    width: 100%;
    max-width: calc(100% - 3rem);
    text-align: center;
    font-weight: bold;
}

/* Model selection */
.model-selection {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1.5rem;
    background: #e6f0fa;
    padding: 1rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    font-weight: bold;
    color: var(--brand-blue);
}
.model-selection label {
    margin-right: 1rem;
    font-size: 1.1rem;
}
.model-selection select {
    padding: 0.5rem;
    border: 1px solid #ccc;
    border-radius: var(--radius);
    background: white;
    cursor: pointer;
    font-size: 1rem;
}

/* Chat message styles */
.chat-message-row {
    display: flex;
    margin-bottom: 1rem;
    animation: fadeIn 0.5s ease-in;
}
.chat-message-row.user { justify-content: flex-end; }
.chat-message-row.bot { justify-content: flex-start; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.user-message, .bot-message {
    padding: 0.75rem 1.25rem;
    border-radius: 1rem;
    max-width: 98%;
    line-height: 1.5;
    font-size: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.user-message {
    background-color: var(--brand-blue);
    color: var(--text-light);
    border-bottom-right-radius: 0.25rem;
}
.bot-message {
    background-color: var(--text-light);
    color: var(--text-dark);
    border: 1px solid #e5e7eb;
    border-bottom-left-radius: 0.25rem;
}

/* Styles for bot response content */
.bot-message p { margin-bottom: 0.75rem; }
.bot-message ul { margin-left: 1.5rem; list-style-type: disc; margin-bottom: 0.75rem; }
.query-details-block {
    border-left: 4px solid #ccc;
    padding-left: 1rem;
    margin-top: 1.25rem;
    font-size: 0.9rem;
    color: #4B5563;
}

/* Table styles within bot messages (CSS Grid Version) */
.grid-table-container {
    overflow-x: auto; /* Provides a horizontal scrollbar if the grid is too wide */
    border: 1px solid #e0e0e0;
    border-radius: var(--radius);
    max-height: 350px;
    overflow-y: auto;
}

.grid-table {
    display: grid; /* This is the magic! */
    /* The column layout is set in the inline style from JavaScript */
}

.grid-header {
    padding: 0.75rem 1rem;
    background-color: var(--brand-blue);
    color: var(--text-light);
    font-weight: bold;
    font-size: 0.9rem;
    position: sticky; /* Makes headers stick to the top on scroll */
    top: 0;
    z-index: 1;
}

.grid-cell {
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
    border-bottom: 1px solid #e0e0e0;
    border-right: 1px solid #e0e0e0;
    background-color: #fff;
    /* This allows long text like sequences to wrap correctly */
    overflow-wrap: break-word;
    min-width: 0; /* Prevents overflow issues with flex/grid items */
}

/* Download link style */
a.download-link {
    display: inline-block;
    background-color: var(--brand-blue);
    color: var(--text-light);
    padding: 0.5rem 1rem;
    border-radius: var(--radius);
    text-decoration: none;
    margin-top: 0.75rem;
    transition: var(--transition);
}
a.download-link:hover {
    background-color: #0055cc;
    transform: translateY(-2px);
}

/* Loading indicator */
.loading-container {
    text-align: center;
    padding: 1rem;
}
.typing-indicator {
    display: flex;
    gap: 0.5rem;
    padding: 1rem;
    justify-content: center;
}
.typing-indicator span {
    width: 10px;
    height: 10px;
    background-color: var(--brand-blue);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

/* Input form styles */
.chat-form {
    display: flex;
    position: relative;
    margin-bottom: 1rem;
    align-items: center;
}
.chat-input {
    flex-grow: 1;
    border: 1px solid #ccc;
    border-radius: 50px;
    padding: 1rem 4rem 1rem 1.5rem;
    font-size: 1rem;
    transition: var(--transition);
}
.chat-input:focus {
    outline: none;
    border-color: var(--brand-blue);
    box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
}
.chat-form button[type="submit"] {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    background-color: var(--brand-blue);
    color: var(--text-light);
    border: none;
    border-radius: 50%;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 1.5rem;
    line-height: 1;
    transition: var(--transition);
}
.chat-form button[type="submit"]:hover {
    background-color: #0055cc;
    transform: translateY(-50%) scale(1.05);
}
.chat-form button[type="submit"]:disabled {
    background-color: #a9a9a9;
    cursor: not-allowed;
    transform: translateY(-50%);
}

/* Secondary button style */
.btn.danger {
    width: 100%;
    padding: 0.75rem;
    font-weight: bold;
    border-radius: var(--radius);
    background-color: transparent;
    color: #a94442;
    border: 1px solid #ebccd1;
    transition: var(--transition);
}
.btn.danger:hover {
    background-color: #f2dede;
    color: #a94442;
}
.btn.danger:disabled {
    background-color: #f5f5f5;
    color: #ccc;
    border-color: #ccc;
    cursor: not-allowed;
}

/* Welcome message */
.chat-welcome {
    text-align: center;
    color: #000;
    width: 100%;
    max-width: 500px;
    margin: 2rem auto;
}

/* Sample Questions */
.sample-questions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
    justify-content: center;
    font-weight: bold;
}
.sample-question-btn {
    background-color: lightblue;
    color: var(--text-dark);
    border: none;
    border-radius: var(--radius);
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    cursor: pointer;
    transition: var(--transition);
    white-space: nowrap;
}
.sample-question-btn:hover {
    background-color: var(--light-blue);
    transform: translateY(-2px);
}
.sample-question-btn:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

/* Error message */
.error-box {
    margin-bottom: 1.5rem;
    padding: 1rem;
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
    border-radius: var(--radius);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.error-box button {
    background: none;
    border: none;
    cursor: pointer;
    color: inherit;
    font-size: 1.2rem;
}

/* API Key Modal Styles */
.api-key-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2000;
    padding: 1rem;
}

.api-key-modal-content {
    background: white;
    padding: 2rem;
    border-radius: var(--radius);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    width: 90%;
    max-width: 600px;
    max-height: 90vh;
    overflow-y: auto;
    text-align: center;
}

.api-key-modal-header {
    margin-bottom: 1.5rem;
}

.api-key-modal-header h2 {
    color: var(--brand-blue);
    margin-bottom: 0.5rem;
    font-size: 1.8rem;
}

.api-key-modal-intro {
    color: var(--text-dark);
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

.api-key-guide {
    background-color: #f8f9fa;
    border-left: 4px solid var(--brand-blue);
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-radius: 0 0.75rem 0.75rem 0;
    text-align: left;
}

.api-key-guide h3 {
    color: var(--brand-blue);
    margin-bottom: 1rem;
    font-size: 1.3rem;
}

.api-key-guide ol {
    margin: 0;
    padding-left: 1.5rem;
}

.api-key-guide li {
    margin-bottom: 0.75rem;
    color: var(--text-dark);
}

.api-key-guide strong {
    color: var(--mid-blue);
}

.guide-tip {
    background-color: #e3f2fd;
    border-left: 3px solid var(--light-blue);
    padding: 0.75rem;
    margin: 1rem 0;
    border-radius: 0 0.5rem 0.5rem 0;
    font-style: italic;
    color: var(--mid-blue);
}

.api-key-form {
    margin: 1.5rem 0;
}

.api-key-input-group {
    margin-bottom: 1rem;
}

.api-key-input-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 600;
    color: var(--brand-blue);
}

.api-key-input {
    width: 100%;
    padding: 0.75rem;
    border: 2px solid #e0e0e0;
    border-radius: var(--radius);
    font-size: 1rem;
    font-family: monospace;
    letter-spacing: 0.05em;
    transition: var(--transition);
}

.api-key-input:focus {
    outline: none;
    border-color: var(--brand-blue);
    box-shadow: 0 0 0 3px rgba(0, 76, 151, 0.1);
}

.api-key-submit-btn {
    width: 100%;
    padding: 0.75rem 1.5rem;
    border: none;
    background-color: var(--brand-blue);
    color: white;
    border-radius: var(--radius);
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    transition: var(--transition);
}

.api-key-submit-btn:hover:not(:disabled) {
    background-color: #0055cc;
    transform: translateY(-1px);
}

.api-key-submit-btn:disabled {
    background-color: #a9a9a9;
    cursor: not-allowed;
}

.api-key-help {
    margin-top: 1.5rem;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: var(--radius);
    font-size: 0.9rem;
}

.api-key-help a {
    color: var(--mid-blue);
    text-decoration: underline;
}

.api-key-help a:hover {
    color: var(--light-blue);
}

/* ===== RESPONSIVE LAYOUT ===== */
@media (max-width: 1200px) {
    .main-container {
        width: calc(100% - 2rem);
        gap: 1.5rem;
    }
    .sidebar {
        flex: 0 0 220px;
        padding: 1rem;
    }
    h1 { font-size: 2.2rem; }
    h2 { font-size: 1.8rem; }
}
@media (max-width: 992px) {
    .main-container {
        flex-direction: column;
        align-items: stretch;
        width: 100%;
    }
    .sidebar {
        position: relative;
        top: auto;
        flex: unset;
        width: 100%;
        margin-bottom: 1rem;
    }
    .chat-center {
        order: -1;
    }
    nav a {
        margin-right: 20px;
        text-decoration: none;
        color: var(--text-light);
        font-size: 1rem;
        font-weight: 500;
        transition: var(--transition);
    }
    nav a:last-child {
        margin-right: 0;
    }
    nav a:hover { opacity: 0.8; }
}
@media (max-width: 768px) {
    .navbar .container {
        flex-direction: column;
        align-items: flex-start;
    }
    nav {
        margin-top: 0.75rem;
    }
    nav a {
        display: inline-block;
        margin: 0.5rem 1rem 0 0;
        font-size: 1rem;
        transition: var(--transition);
    }
    nav a:last-child {
        margin-right: 0;
    }
    nav a:hover { opacity: 0.8; }
    .brand {
        font-size: 1.5rem;
    }
    .logo-circle {
        width: 45px;
        height: 45px;
    }
    .chat-input {
        font-size: 0.95rem;
        padding: 0.75rem 3.5rem 0.75rem 1rem;
    }
    .chat-form button[type="submit"] {
        width: 38px;
        height: 38px;
        font-size: 1.2rem;
    }
    .api-key-modal-content {
        width: 95%;
        padding: 1.5rem;
    }
}
@media (max-width: 480px) {
    h1 { font-size: 1.8rem; }
    h2 { font-size: 1.5rem; }
    h3 { font-size: 1.2rem; }
    .chat-wrapper {
        padding: 1rem;
    }
#chat-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

    .user-message, .bot-message {
        font-size: 0.9rem;
        max-width: 90%;
    }
    .bot-intro {
        font-size: 0.9rem;
    }
    .sample-question-btn {
        font-size: 0.8rem;
        padding: 0.4rem 0.8rem;
    }
    .api-key-modal-content {
        padding: 1rem;
    }
    .api-key-guide {
        padding: 1rem;
    }
}
</style>
<script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.development.js"></script>
<script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.development.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.22.5/babel.min.js"></script>
</head>
<body>
<!-- Include Header -->
<?php include 'header.html'; ?>
<!-- ===== MAIN CONTENT ===== -->
<div class="main-container">
<!-- Left Sidebar: Database Papers -->
<aside class="sidebar left-sidebar">
<h3><b>Database Papers</b></h3>
<div id="db-papers"></div>
</aside>
<!-- Center: Chat -->
<main class="chat-center">
<h1 id="chatbot-title" class="welcome-title">Chatbot</h1>
<div id="root"></div>
</main>
<!-- Right Sidebar: Notifications and Model Links -->
<aside class="sidebar right-sidebar">
<h3><b>Updates</b></h3>
<ul id="bot-updates-list"></ul>
</aside>
</div>
<!-- ===== REACT SCRIPT ===== -->
<script type="text/babel">
const { useState, useEffect, useRef } = React;

const dbPapers = {
    'trf': [
        { url: 'https://academic.oup.com/database/article/doi/10.1093/database/bay063/5043071?login=true', img: 'images/databases/ptrfdb_1.jpeg', name: 'PtRFdb: a database for plant transfer RNA-derived fragments Open Access' }
    ],
    'fusion': [
        { url: 'https://academic.oup.com/database/article/doi/10.1093/database/bay135/5277248?login=true', img: 'images/databases/atfusiondb_1.jpeg', name: 'AtFusionDB: a database of fusion transcripts in Arabidopsis thaliana' },
        { url: 'https://link.springer.com/article/10.1007/s13205-024-04132-1', img: 'images/databases/pfusiondb_1.jpg', name: 'PFusionDB: a comprehensive database of plant-specific fusion transcripts' }
    ],
    'amp': [
        { url: 'https://www.nature.com/articles/s41598-020-59165-2', img: 'images/databases/plantpep_1.jpg', name: 'PlantPepDB: A manually curated plant peptide database' }
    ],
    'cotton': [
        { url: 'https://www.sciencedirect.com/science/article/pii/S2001037023001939?via%3Dihub', img: 'images/databases/concratlas_1.jpg', name: 'Long non-coding RNA and microRNA landscape of two major domesticated cotton species' }
    ],
    'ncrna': [
        { url: 'https://link.springer.com/article/10.1007/s13205-022-03174-7', img: 'images/databases/ptncrnadb_1.jpg', name: 'PtncRNAdb: plant transfer RNA-derived non-coding RNAs (tncRNAs) database' }
    ],
    'pvsi': [
        { url: 'https://doi.org/10.1093/database/bay105', img: 'images/databases/pvsi.jpeg', name: 'PVsiRNAdb: a database for plant exclusive virus-derived small interfering RNAs ' }
    ],
    'atfusion': [
        { url: 'https://academic.oup.com/database/article/doi/10.1093/database/bay135/5277248?login=true', img: 'images/databases/atfusiondb_1.jpeg', name: 'AtFusionDB: a database of fusion transcripts in Arabidopsis thaliana' }
    ],
    'pfusion': [
        { url: 'https://link.springer.com/article/10.1007/s13205-024-04132-1', img: 'images/databases/pfusiondb_1.jpg', name: 'PFusionDB: a comprehensive database of plant-specific fusion transcripts' }
    ],
    'ptrfbot': [
        { url: 'https://academic.oup.com/database/article/doi/10.1093/database/bay063/5043071?login=true', img: 'images/databases/ptrfdb_1.jpeg', name: 'PtRFdb: a database for plant transfer RNA-derived fragments Open Access' }
    ],
    'pbtrfbot': [
        { url: 'https://link.springer.com/article/10.1007/s10142-025-01576-3', img: 'images/databases/pbtrf_1.jpg', name: 'Comprehensive study of tRNA-derived fragments in plants for biotic stress responses' }
    ],
    'ptncbot': [
        { url: 'https://link.springer.com/article/10.1007/s13205-022-03174-7', img: 'images/databases/ptncrnadb_1.jpg', name: 'PtncRNAdb: plant transfer RNA-derived non-coding RNAs (tncRNAs) database' }
    ],
    'athisomirbot': [
        { url: 'https://academic.oup.com/database/article/doi/10.1093/database/baae115/7887559?login=true', img: 'images/databases/athisomirbot.jpeg', name: 'athisomiRDB: A comprehensive database of Arabidopsis isomiRs' }
    ],
    'anninterbot': [
        { url: 'https://www.sciencedirect.com/science/article/pii/S1476927124003165?via%3Dihub', img: 'images/databases/anninterbot.jpg', name: 'ANNInter: A platform to explore ncRNA-ncRNA interactome of Arabidopsis thaliana' }
    ],
    'alncbot': [
        { url: 'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247215', img: 'images/databases/alncbot.PNG', name: 'AlnC: An extensive database of long non-coding RNAs in angiosperms' }
    ],
    'ptrnabot': [
        { url: 'https://link.springer.com/article/10.1007/s13205-022-03255-7', img: 'images/databases/ptrnabot.jpg', name: 'PtRNAdb: a web resource of plant tRNA genes from a wide range of plant species'}
    ]
};

const botDescriptions = {
    'trf': 'Explore plant tRNA-derived fragments (tRFs), emerging regulators of gene expression, stress adaptation, and epigenetic control, supported by comprehensive datasets for functional and comparative analysis.',
    'fusion': 'Discover fusion transcripts in plants, novel RNAs from gene fusions, with insights from AtFusionDB and PFusionDB.',
    'amp': 'Learn about antimicrobial peptides (AMPs) in plants, critical for pathogen defense, using PlantPepDB data.',
    'cotton': 'Investigate long non-coding RNAs and microRNAs in cotton, vital for fiber quality and stress responses, via ConCRAtlas.',
    'ncrna': 'Dive into tRNA-derived non-coding RNAs (tncRNAs) in plants, focusing on their regulatory roles, using PtncRNAdb.',
    'pvsi': 'Explore virus-derived small interfering RNAs (vsiRNAs) in plants, key to antiviral defense, with PVsiRNAdb data.',
    'atfusion': 'Study fusion transcripts in Arabidopsis thaliana, their roles and validation, using AtFusionDB.',
    'pfusion': 'Investigate plant-specific fusion transcripts, their traits and evolution, with PFusionDB data.',
    'ptrfbot': 'Explore tRNA-derived fragments in plants, focusing on biotic stress responses, using PtRFdb.',
    'pbtrfbot': 'Learn about tRNA-derived fragments in plants under biotic stress, with insights from PbtRF database.',
    'ptncbot': 'Discover tRNA-derived non-coding RNAs in monocots and other plants, using PtncRNAdb.',
    'athisomirbot': 'Investigate isomiRs in Arabidopsis, microRNA variants, and their roles, using athisomiRDB.',
    'anninterbot': 'Explore ncRNA-ncRNA interactions in Arabidopsis, their predictions and roles, via ANNInter.',
    'alncbot': 'Study long non-coding RNAs (lncRNAs) in angiosperms, their development and evolution, using AlnC.',
    'ptrnabot': 'Learn about tRNA genes in plants, their annotation and roles in translation, via PtRNAdb.'
};

const sampleQuestions = {
    'trf': [
        'What are trfs/tnc RNAs?',
        'How do trfs are formed?',
        'What is the function for trfs?',
        'What are the functions of tRNA-derived fragments in stress?',
        'How many tRF-3 fragments are present in Oryza sativa, and what are their sequence lengths?',
        'Which tRFs respond under heat stress?'

    ],
    'fusion': [
        'What are fusion transcripts?',
        'What is the role of fusion transcripts in plants?',
        'How are fusion transcripts generated?',
        'What information on fusion transcripts do you have?',
        'Do AT3G11170 & AT5G05580 make any fusions?',
        'Show me fusion transcripts in Arabidopsis thaliana.',
        'How are fusion transcripts detected?',
        'What unique fusion transcripts are found in Arabidopsis thaliana, including their parental genes and sequences?'
    ],
    'amp': [
        'What are therapeutic peptides?',
        'What are antimicrobial peptides in plants?',
        'List peptides in PlantPepDB.',
        'What is the role of peptides in plant defense?',
        'what peptides do you have that have antibacterial properties?',
        'List peptides having both Antibacterial & Anticancer activity?'
    ],
    'cotton': [
        'What are long non-coding RNAs in cotton?',
        'Show microRNA data for cotton species.',
        'How do lncRNAs affect cotton stress response?',
        'what is a lncRNA?',
        'Do lncRNA have any role in plant stress?',
        'Can you provide sequences and processing information for specific miRNAs by their IDs in cotton?',
        'List lncRNAs targeted by miRNAs on chromosome 1',
        'List miRNA and lncRNA having high expression in leaf tissue.'
    ],
    'ncrna': [
        'What are non-coding RNAs in plants?',
        'How ncRNA help plants to response against heat stess?',
        'List tncRNAs in PtncRNAdb.',
        'How do tncRNAs regulate plant gene expression?',
        'Fetch me interactions for the following miRNA UGACAGAAGAGAGUGAGCAC',
        'Retrieve isomiRs with high RPM in specific tissues.',
    ],
    'pvsi': [
        'What are virus-derived small interfering RNAs in plants?',
        'List vsiRNAs in PVsiRNAdb.',
        'How do vsiRNAs contribute to plant antiviral defense?'
    ],
    'atfusion': [
        'What fusion transcripts are available in AtFusionDB?',
        'Show fusion events in Arabidopsis.',
        'How are fusion transcripts validated?'
    ],
    'pfusion': [
        'List plant-specific fusion transcripts in PFusionDB.',
        'What are the sources of fusion transcripts?',
        'How do fusion transcripts affect plant traits?'
    ],
    'ptrfbot': [
        'What are tRNA-derived fragments in PtRFdb?',
        'Show tRFs associated with biotic stress.',
        'How are tRFs sequenced in plants?'
    ],
    'pbtrfbot': [
        'How do tRFs respond to biotic stress in plants?',
        'List tRFs in PbtRF database.',
        'What is the role of tRFs in plant immunity?'
    ],
    'ptncbot': [
        'What are tncRNAs in PtncRNAdb?',
        'Show tncRNAs in monocots.',
        'How do tncRNAs function in plants?'
    ],
    'athisomirbot': [
        'What are isomiRs in Arabidopsis?',
        'List isomiRs in athisomiRDB.',
        'How do isomiRs regulate gene expression?'
    ],
    'anninterbot': [
        'What ncRNA interactions are in ANNInter?',
        'Show ncRNA-ncRNA interactions in Arabidopsis.',
        'How are ncRNA interactions predicted?'
    ],
    'alncbot': [
        'What are long non-coding RNAs in angiosperms?',
        'List lncRNAs in AlnC database.',
        'How do lncRNAs regulate plant development?'
    ],
    'ptrnabot': [
        'What are tRNA genes in PtRNAdb?',
        'List tRNA genes in rice.',
        'How are tRNA genes annotated in plants?'
    ]
};

const ApiKeyModal = ({ show, onApiKeySubmit }) => {
    if (!show) return null;

    const [apiKey, setApiKey] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!apiKey.trim()) return;
        
        setIsGenerating(true);
        try {
            onApiKeySubmit(apiKey.trim());
        } finally {
            setIsGenerating(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <div className="api-key-modal">
            <div className="api-key-modal-content">
                <div className="api-key-modal-header">
                    <h2>🔑 Setup Your Groq API Key</h2>
                    <p className="api-key-modal-intro">
                        To unlock the full power of our plant regulatory elements chatbot, you'll need a Groq API key. 
                        This free key allows you to access state-of-the-art AI models for your research queries.
                    </p>
                </div>

                {/* Step-by-step guide */}
                <div className="api-key-guide">
                    <h3>📋 How to Generate Your Free API Key</h3>
                    <ol>
                        <li>
                            <strong>Visit the Groq Console:</strong> Click the button below or go to{' '}
                            <a href="https://console.groq.com/" target="_blank" rel="noopener noreferrer">
                                console.groq.com
                            </a>
                        </li>
                        <li>
                            <strong>Create an Account:</strong> If you don't have an account, click "Sign Up" and 
                            complete the registration process (it's free and quick).
                        </li>
                        <li>
                            <strong>Verify Your Email:</strong> Check your inbox for a verification email from Groq 
                            and click the verification link.
                        </li>
                        <li>
                            <strong>Navigate to API Keys:</strong> Once logged in, look for "API Keys" in the left sidebar 
                            or click the "Keys" tab at the top.
                        </li>
                        <li>
                            <strong>Create New Key:</strong> Click the "Create API Key" button.
                        </li>
                        <li>
                            <strong>Name Your Key:</strong> Give it a descriptive name like "Plant Regulatory Elements Chatbot".
                        </li>
                        <li>
                            <strong>Copy the Key:</strong> Your new API key will be generated. Copy it immediately 
                            (you won't be able to see it again after closing the dialog).
                        </li>
                    </ol>
                    
                    <div className="guide-tip">
                        💡 <strong>Pro Tip:</strong> The API key format starts with "gsk_". Keep it secure and never share it publicly. 
                        You can always generate new keys in the console if needed.
                    </div>
                </div>

                {/* API Key Input Form */}
                <form onSubmit={handleSubmit} className="api-key-form">
                    <div className="api-key-input-group">
                        <label htmlFor="apiKeyInput">Enter Your API Key:</label>
                        <input
                            type="password"
                            id="apiKeyInput"
                            name="apiKey"
                            value={apiKey}
                            onChange={(e) => setApiKey(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                            className="api-key-input"
                            required
                            disabled={isGenerating}
                            autoComplete="off"
                        />
                        <small style={{color: '#666', fontSize: '0.85rem', marginTop: '0.25rem', display: 'block'}}>
                            Your key is encrypted and stored only for this session. It will be cleared when you close the browser.
                        </small>
                    </div>
                    
                    <button 
                        type="submit" 
                        className="api-key-submit-btn"
                        disabled={!apiKey.trim() || isGenerating}
                    >
                        {isGenerating ? '🔄 Connecting...' : '🚀 Start Chatting with AI'}
                    </button>
                </form>

                {/* Quick Access Links */}
                <div className="api-key-help">
                    <p><strong>Need help?</strong></p>
                    <p>
                        <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer">
                            🔗 Direct Link: Create API Key
                        </a>
                    </p>
                    <p>
                        <a href="https://docs.groq.com/" target="_blank" rel="noopener noreferrer">
                            📚 Groq Documentation
                        </a>
                    </p>
                    <p style={{fontSize: '0.8rem', color: '#666', marginTop: '0.5rem'}}>
                        <strong>Free Tier:</strong> Includes generous usage limits for research and development.
                    </p>
                </div>
            </div>
        </div>
    );
};

const ChatApp = () => {
    const [query, setQuery] = useState('');
    const [conversation, setConversation] = useState([]);
    const [conversationId, setConversationId] = useState(() => crypto.randomUUID ? crypto.randomUUID() : Date.now().toString());
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [model, setModel] = useState('llama-3.3-70b-versatile'); // Default to Llama model
    const chatContainerRef = useRef(null);
    const [botName, setBotName] = useState('');
    const [botDisplayName, setBotDisplayName] = useState('Chatbot');
    const [apiKey, setApiKey] = useState(null);
    const [showApiKeyModal, setShowApiKeyModal] = useState(true);

    useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const bot = urlParams.get('bot');
        if (bot) {
            setBotName(bot);
            const botDisplayNames = {
                'trf': 'tRF Bot',
                'atfusion': 'AtFusion Bot',
                'fusion': 'Fusion Bot',
                'amp': 'AMP Bot',
                'cotton': 'Cotton Bot',
                'ncrna': 'ncRNA Bot',
                'pvsi': 'Pvsi Bot',
                'pfusion': 'PFusion Bot',
                'ptrfbot': 'PtRF Bot',
                'pbtrfbot': 'PbtRF Bot',
                'ptncbot': 'Ptncdb Bot',
                'ptrnabot': 'PtRNAdb Bot',
                'athisomirbot': 'athisomiR Bot',
                'anninterbot': 'ANNInter Bot',
                'alncbot': 'AlnC Bot'
            };
            const displayName = "Chat with " + (botDisplayNames[bot] || 'Chatbot');
            setBotDisplayName(displayName);
            document.getElementById('chatbot-title').innerText = displayName;
            // Populate DB papers
            const papersContainer = document.getElementById('db-papers');
            papersContainer.innerHTML = '';
            const botPapers = dbPapers[bot] || [];
            botPapers.forEach(paper => {
                const paperElement = document.createElement('a');
                paperElement.href = paper.url;
                paperElement.target = '_blank';
                paperElement.className = 'db-card';
                paperElement.innerHTML = `<img src="${paper.img}" alt="${paper.name}" /><div class="db-name">${paper.name}</div>`;
                papersContainer.appendChild(paperElement);
            });
            // Populate updates dynamically from CSV
            fetch('bot_updates.csv')
                .then(response => response.text())
                .then(csvText => {
                    const lines = csvText.trim().split('\n');
                    const updatesList = lines.slice(1).map(line => {
                        const [csvBot, updateText, link] = line.split(',');
                        return {
                            csvBot: csvBot.trim(),
                            updateText: updateText.trim(),
                            link: link ? link.trim() : null
                        };
                    }).filter(u => u.csvBot === bot);
                    const updatesUl = document.getElementById('bot-updates-list');
                    updatesUl.innerHTML = '';
                    updatesList.forEach(u => {
                        const li = document.createElement('li');
                        li.innerHTML = u.link
                            ? `${u.updateText} <a href="${u.link}" target="_blank">Read more</a>`
                            : u.updateText;
                        updatesUl.appendChild(li);
                    });
                });
        }
    }, [botName]);

    const handleQuestionClick = (question) => {
        setQuery(question);
    };

    const handleApiKeySubmit = (key) => {
        setApiKey(key);
        setShowApiKeyModal(false);
        setError('');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!query.trim() || isLoading || !apiKey) {
            if (!apiKey) {
                setError("Please provide an API key before starting a chat.");
                setShowApiKeyModal(true);
            }
            return;
        }
        setIsLoading(true);
        setError('');
        const newConversation = [...conversation, { type: 'user', content: query }];
        setConversation(newConversation);
        const currentQuery = query;
        setQuery('');
        try {
            const response = await fetch(`/PlantXBot/public/connect.php/query/${botName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: currentQuery,
                    conversation_id: conversationId,
                    model: model,
                    api_key: apiKey
                })
            });
            if (!response.ok) {
                if (response.status === 429) {
                    setError("API rate limit reached. Please provide a new API key or wait for the limit to reset.");
                    setShowApiKeyModal(true);
                }
                const errorText = await response.text();
                let errorDetail = errorText;
                try { const errorJson = JSON.parse(errorText); errorDetail = errorJson.error || errorJson.message || errorText; } catch {}
                throw new Error(`HTTP error ${response.status}: ${errorDetail}`);
            }
            const data = await response.json();
            if (data.error && typeof data.error === 'string') {
                throw new Error(data.error);
            }
            const botMessageHtml = formatBotResponse(data);
            setConversation(prev => [...prev, { type: 'bot', content: botMessageHtml }]);
        } catch (error) {
            console.error('Fetch error:', error);
            const errorMessage = error.message || 'Failed to fetch response.';
            setError(errorMessage);
            setConversation(prev => [...prev, { type: 'bot', content: `<p style="color: red;">Error: ${errorMessage}</p>`, isError: true }]);
        }
        setIsLoading(false);
    };

    const clearChat = () => {
        setConversation([]);
        setQuery('');
        setError('');
        setConversationId(crypto.randomUUID ? crypto.randomUUID() : Date.now().toString());
    };

const formatBotResponse = (data) => {
        let messageParts = [];
        if (data.summary) messageParts.push(`<p>${data.summary}</p>`);
        else if (data.error) messageParts.push(`<p style="color: red;">Error from Bot: ${data.error}</p>`);
        else messageParts.push(`<p>No summary provided.</p>`);

        if (data.executed_queries_details && data.executed_queries_details.length > 0) {
            data.executed_queries_details.forEach(queryDetail => {
                const tableData = queryDetail.results_preview || [];
                if (tableData.length > 0) {
                    let detailBlock = `<div class="query-details-block">`;
                    if (queryDetail.description) detailBlock += `<p style="font-style: italic; font-size: 0.9em;">Purpose: ${queryDetail.description}</p>`;
                    if (queryDetail.error) detailBlock += `<p style="color: red; font-size: 0.9em;">Error for this part: ${queryDetail.error}</p>`;
                    else {
                        // --- START: NEW GRID TABLE GENERATION ---
                        detailBlock += `<p style="margin-top: 0.5rem; margin-bottom: 0.5rem; font-weight: 500; font-size: 0.9em;">Data Preview:</p>`;
                        
                        // Get all unique headers from all rows
                        const allHeaders = [...new Set(tableData.flatMap(row => Object.keys(row)))];
                        
                        // Create the main grid container
                        // This dynamically sets the number of columns.
                        detailBlock += `<div class="grid-table-container">
                                            <div class="grid-table" style="grid-template-columns: repeat(${allHeaders.length}, minmax(160px, auto));">`;

                        // Add header cells
                        allHeaders.forEach(header => {
                            detailBlock += `<div class="grid-header">${header.replace(/_/g, ' ')}</div>`;
                        });

                        // Add data cells
                        tableData.forEach(row => {
                            allHeaders.forEach(header => {
                                const cellValue = row[header] === null || row[header] === undefined ? '' : String(row[header]);
                                detailBlock += `<div class="grid-cell" title="${cellValue.replace(/"/g, '&quot;')}">${cellValue}</div>`;
                            });
                        });
                        
                        detailBlock += `</div></div>`;
                        // --- END: NEW GRID TABLE GENERATION ---
                    }
                    if (queryDetail.download_url) detailBlock += `<p><a href="${queryDetail.download_url}" class="download-link" target="_blank" rel="noopener noreferrer">Download Full CSV</a></p>`;
                    detailBlock += `</div>`;
                    messageParts.push(detailBlock);
                }
            });
        }
        if (data.metadata && data.metadata.execution_time) {
            messageParts.push(`<div style="margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid #eee; font-size: 0.8em; color: #888;"><p>Time: ${data.metadata.execution_time}s</p></div>`);
        }
        return messageParts.join('');
    };

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [conversation]);

    return (
        <div className="chat-wrapper">
            <ApiKeyModal show={showApiKeyModal} onApiKeySubmit={handleApiKeySubmit} />
            <>
                <div className="bot-intro">
                    {botDescriptions[botName] || 'Explore data related to plant regulatory elements using this chatbot.'}
                </div>
                <div className="model-selection">
                    <label htmlFor="model-select">Choose a Model:</label>
                    <select id="model-select" value={model} onChange={(e) => setModel(e.target.value)}>
                        <option value="llama-3.3-70b-versatile">🐪 Llama-3.3-70B-Versatile (Recommended)</option>
                        <option value="openai/gpt-oss-20b">🤖 GPT-OSS-20B</option>
                    </select>
                </div>
                {error && (
                    <div className="error-box" role="alert">
                        <span>{error}</span>
                        <button onClick={() => setError('')} aria-label="Dismiss error">×</button>
                    </div>
                )}
                <div id="chat-container" ref={chatContainerRef}>
                    {conversation.length === 0 && !isLoading && (
                        <div className="chat-welcome">
                            <p style={{ fontSize: '1.25rem' }}>Welcome to the {botDisplayName}!</p>
                            <div className="sample-questions">
                                {sampleQuestions[botName]?.length > 0 ? (
                                    sampleQuestions[botName].map((question, index) => (
                                        <button
                                            key={index}
                                            className="sample-question-btn"
                                            onClick={() => handleQuestionClick(question)}
                                        >
                                            {question}
                                        </button>
                                    ))
                                ) : (
                                    <p>No sample questions available.</p>
                                )}
                            </div>
                        </div>
                    )}
                    {conversation.map((msg, index) => (
                        <div key={index} className={`chat-message-row ${msg.type}`}>
                            <div className={`${msg.type}-message`}>
                                <p style={{ fontWeight: 600, marginBottom: '0.25rem' }}>{msg.type === 'user' ? 'You' : 'Bot'}:</p>
                                <div dangerouslySetInnerHTML={{ __html: msg.content }} />
                            </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="loading-container">
                            <div className="typing-indicator"><span></span><span></span><span></span></div>
                        
                        </div>
                    )}
                </div>
                <form onSubmit={handleSubmit} className="chat-form">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask a question..."
                        className="chat-input"
                        aria-label="Query input"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !query.trim()}
                        className="btn primary"
                        aria-label="Submit query"
                    >
                        &#10148;
                    </button>
                </form>
                <button
                    onClick={clearChat}
                    className="btn danger"
                    aria-label="Clear chat"
                    disabled={isLoading}
                >
                    Clear & Reset
                </button>
            </>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><ChatApp /></React.StrictMode>);
</script>
<!-- Include Footer -->
<?php include 'footer.html'; ?>
</body>
</html>


