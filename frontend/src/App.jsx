import { useEffect, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function App() {
  const [status, setStatus] = useState('checking...')

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then((data) => setStatus(data.status))
      .catch(() => setStatus('backend unreachable'))
  }, [])

  return (
    <main className="page">
      <h1>CallSense AI</h1>
      <p>Customer Conversation Analytics &amp; Quality Intelligence Platform</p>
      <p className="status">
        Backend status: <strong>{status}</strong>
      </p>
      <p className="note">
        Dashboard modules (call analysis, risk view, agent analytics, search) land
        here module by module — see <code>docs/module_01_problem_definition_and_architecture.md</code>.
      </p>
    </main>
  )
}

export default App
