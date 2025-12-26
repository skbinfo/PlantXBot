<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChatBot | Data Intelligence Platform</title>
<!-- Replace with your favicon -->
<link rel="icon" href="images/favicon.png" type="image/png">
<style>
/* ===== GLOBAL & TYPOGRAPHY ===== */
:root {
    /* Branding Colors */
    --brand-blue: #2c3e50; /* Primary */
    --mid-blue:   #34495e; /* Secondary */
    --light-blue: #3498db; /* Accent */
    
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
h3 { font-size: 1.5rem; color: var(--mid-blue); margin-bottom: 1rem; }

/* ===== LAYOUT & SIDEBARS ===== */
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
.chat-center {
    flex: 1;
    min-width: 0;
}

/* Sidebar Item Styles */
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
    height: 100px;
    object-fit: cover;
    border-bottom: 1px solid #e0e0e0;
}
.left-sidebar .db-name {
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: var(--brand-blue);
    padding: 0.5rem;
    text-align: center;
    font-weight: 600;
}

/* Chat Component Styles */
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
    height: 500px;
    overflow-y: auto;
    scroll-behavior: smooth;
    border: 1px solid #e0e0e0;
    border-radius: var(--radius);
    padding: 1.5rem;
    background-color: #fdfdfd;
}

/* Message Bubbles */
.chat-message-row {
    display: flex;
    margin-bottom: 1rem;
    animation: fadeIn 0.3s ease-in;
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
    max-width: 85%;
    line-height: 1.5;
    font-size: 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.user-message {
    background-color: var(--brand-blue);
    color: var(--text-light);
    border-bottom-right-radius: 0.25rem;
}
.bot-message {
    background-color: var(--gray);
    color: var(--text-dark);
    border: 1px solid #e5e7eb;
    border-bottom-left-radius: 0.25rem;
}

/* Grid Table for Data Preview */
.grid-table-container {
    overflow-x: auto;
    border: 1px solid #ddd;
    border-radius: var(--radius);
    max-height: 300px;
    margin-top: 10px;
    background: #fff;
}
.grid-table {
    display: grid; 
    /* Columns defined dynamically in JS */
}
.grid-header {
    background-color: var(--mid-blue);
    color: #fff;
    padding: 8px;
    font-weight: bold;
    font-size: 0.85rem;
    position: sticky;
    top: 0;
}
.grid-cell {
    padding: 8px;
    font-size: 0.85rem;
    border-bottom: 1px solid #eee;
    border-right: 1px solid #eee;
    overflow-wrap: break-word;
}

/* Inputs & Buttons */
.chat-form {
    display: flex;
    position: relative;
    margin-top: 1rem;
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
    box-shadow: 0 0 0 3px rgba(44, 62, 80, 0.2);
}
.chat-form button[type="submit"] {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    background-color: var(--brand-blue);
    color: white;
    border: none;
    border-radius: 50%;
    width: 44px;
    height: 44px;
    cursor: pointer;
    font-size: 1.2rem;
}

/* API Key Modal */
.api-key-modal {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2000;
}
.api-key-modal-content {
    background: white;
    padding: 2rem;
    border-radius: var(--radius);
    max-width: 500px;
    width: 90%;
    text-align: center;
}
.api-key-input {
    width: 100%;
    padding: 0.75rem;
    margin: 1rem 0;
    border: 2px solid #ddd;
    border-radius: var(--radius);
}
.api-key-submit-btn {
    background-color: var(--brand-blue);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: var(--radius);
    cursor: pointer;
    width: 100%;
    font-weight: bold;
}

/* Responsive */
@media (max-width: 992px) {
    .main-container { flex-direction: column; }
    .sidebar { width: 100%; margin-bottom: 1rem; }
}
</style>
<script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.development.js"></script>
<script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.development.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.22.5/babel.min.js"></script>
</head>
<body>

<?php include 'header.html'; ?>

<div class="main-container">
    <!-- Left Sidebar: Context / Documents -->
    <aside class="sidebar left-sidebar">
        <h3><b>Related Datasets</b></h3>
        <div id="db-papers"></div>
    </aside>

    <!-- Center: Chat Interface -->
    <main class="chat-center">
        <h1 id="chatbot-title">Generic Chatbot</h1>
        <div id="root"></div>
    </main>

    <!-- Right Sidebar: Updates / Notifications -->
    <aside class="sidebar right-sidebar">
        <h3><b>System Updates</b></h3>
        <ul id="bot-updates-list">
            <li>System maintenance scheduled for Sunday.</li>
            <li>New dataset "Q3 Financials" added.</li>
        </ul>
    </aside>
</div>

<!-- React Logic -->
<script type="text/babel">
const { useState, useEffect, useRef } = React;

// --- DUMMY DATA CONFIGURATION ---
// Map bot 'slugs' (from URL ?bot=slug) to human readable names and context
const botConfig = {
    'demo': {
        name: 'Demo Bot',
        description: 'A generic demo bot connected to the sample database.',
        papers: [
            { name: 'Sample Schema Doc', url: '#', img: 'https://via.placeholder.com/300x150?text=Schema+Doc' },
            { name: 'User Guide', url: '#', img: 'https://via.placeholder.com/300x150?text=User+Guide' }
        ],
        questions: [
            'Show me the first 5 rows of the main inventory',
            'List all categories in the database',
            'What is the total count of items?',
            'Find items with value > 50'
        ]
    },
    'sales': {
        name: 'Sales Analytics',
        description: 'Analyze revenue, leads, and pipeline performance.',
        papers: [
            { name: 'Q3 Report', url: '#', img: 'https://via.placeholder.com/300x150?text=Q3+Report' }
        ],
        questions: ['Total revenue for Q3?', 'List top 5 sales reps']
    }
};

// --- API KEY MODAL COMPONENT ---
const ApiKeyModal = ({ show, onApiKeySubmit }) => {
    if (!show) return null;
    const [key, setKey] = useState('');

    return (
        <div className="api-key-modal">
            <div className="api-key-modal-content">
                <h2>Authentication Required</h2>
                <p>Please enter your API Key (e.g., Groq) to access the chatbot.</p>
                <form onSubmit={(e) => { e.preventDefault(); onApiKeySubmit(key); }}>
                    <input 
                        type="password" 
                        className="api-key-input" 
                        placeholder="gsk_..." 
                        value={key}
                        onChange={e => setKey(e.target.value)}
                        required
                    />
                    <button type="submit" className="api-key-submit-btn">Start Session</button>
                </form>
                <p style={{fontSize: '0.8rem', marginTop: '1rem', color: '#666'}}>
                    <a href="https://console.groq.com/keys" target="_blank">Get a free key here</a>
                </p>
            </div>
        </div>
    );
};

// --- MAIN CHAT APP COMPONENT ---
const ChatApp = () => {
    const [query, setQuery] = useState('');
    const [conversation, setConversation] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [botName, setBotName] = useState('demo');
    const [apiKey, setApiKey] = useState(null);
    const [showModal, setShowModal] = useState(true);
    const chatContainerRef = useRef(null);

    // Load Bot Config on Mount
    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const botParam = params.get('bot') || 'demo';
        setBotName(botParam);

        // Update UI based on config
        const config = botConfig[botParam] || botConfig['demo'];
        document.getElementById('chatbot-title').innerText = config.name;
        
        // Populate Sidebar
        const sidebar = document.getElementById('db-papers');
        sidebar.innerHTML = '';
        config.papers.forEach(p => {
            const el = document.createElement('a');
            el.className = 'db-card';
            el.href = p.url;
            el.innerHTML = `<img src="${p.img}" /><div class="db-name">${p.name}</div>`;
            sidebar.appendChild(el);
        });
    }, []);

    // Scroll to bottom of chat
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [conversation]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!query.trim() || !apiKey) return;

        // Add User Message
        const newHistory = [...conversation, { type: 'user', content: query }];
        setConversation(newHistory);
        setIsLoading(true);
        const currentQ = query;
        setQuery('');

        try {
            // Pointing to our Flask Demo App URL (Port 5001)
            // Ensure demo_app.py is running!
            const res = await fetch(`http://localhost:5001/query/${botName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: currentQ,
                    api_key: apiKey,
                    conversation_id: 'web-session-' + Date.now()
                })
            });

            const data = await res.json();
            
            // Format Response
            const botHtml = formatResponse(data);
            setConversation([...newHistory, { type: 'bot', content: botHtml }]);

        } catch (err) {
            setConversation([...newHistory, { type: 'bot', content: `<p style="color:red">Error: ${err.message}</p>` }]);
        } finally {
            setIsLoading(false);
        }
    };

    const formatResponse = (data) => {
        let html = `<p>${data.summary || data.error || 'No response.'}</p>`;
        
        // Render Tables if available
        if (data.executed_queries_details) {
            data.executed_queries_details.forEach(q => {
                if (q.results_preview && q.results_preview.length > 0) {
                    const headers = Object.keys(q.results_preview[0]);
                    html += `<div class="grid-table-container"><div class="grid-table" style="grid-template-columns: repeat(${headers.length}, minmax(120px, 1fr))">`;
                    
                    // Headers
                    headers.forEach(h => html += `<div class="grid-header">${h}</div>`);
                    
                    // Rows
                    q.results_preview.forEach(row => {
                        headers.forEach(h => html += `<div class="grid-cell">${row[h] || ''}</div>`);
                    });
                    
                    html += `</div></div>`;
                    if(q.download_url) html += `<p><a href="${q.download_url}" target="_blank">Download CSV</a></p>`;
                }
            });
        }
        return html;
    };

    return (
        <div className="chat-wrapper">
            <ApiKeyModal show={showModal} onApiKeySubmit={(k) => { setApiKey(k); setShowModal(false); }} />
            
            <div id="chat-container" ref={chatContainerRef}>
                {conversation.length === 0 && (
                    <div style={{textAlign:'center', marginTop: '2rem'}}>
                        <p>{botConfig[botName]?.description || 'Ready to analyze data.'}</p>
                        <div style={{display:'flex', gap:'10px', justifyContent:'center', flexWrap:'wrap', marginTop:'1rem'}}>
                            {(botConfig[botName]?.questions || []).map((q, i) => (
                                <button key={i} onClick={() => setQuery(q)} style={{padding:'8px', cursor:'pointer', border:'1px solid #ccc', borderRadius:'5px', background:'#eee'}}>
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                )}
                
                {conversation.map((msg, i) => (
                    <div key={i} className={`chat-message-row ${msg.type}`}>
                        <div className={`${msg.type}-message`} dangerouslySetInnerHTML={{ __html: msg.content }} />
                    </div>
                ))}
                
                {isLoading && <div style={{textAlign:'center', color:'#888'}}>Analyzing...</div>}
            </div>

            <form onSubmit={handleSubmit} className="chat-form">
                <input 
                    className="chat-input" 
                    value={query} 
                    onChange={e => setQuery(e.target.value)} 
                    placeholder="Ask a question about the data..." 
                    disabled={isLoading}
                />
                <button type="submit" disabled={isLoading}>&#10148;</button>
            </form>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><ChatApp /></React.StrictMode>);
</script>

<?php include 'footer.html'; ?>
</body>
</html>
