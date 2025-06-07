const { app, BrowserWindow, ipcMain, shell } = require('electron')
const path = require('path')
const { spawn } = require('child_process')

let mainWindow
let backendProcess

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false,
    },
    icon: path.join(__dirname, '../public/favicon.ico'),
    show: false,
  })

  const isDev = process.env.NODE_ENV === 'development'

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173')
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
}

function startBackend() {
  let backendPath
  
  const possiblePaths = []
  
  if (process.platform === 'win32') {
    possiblePaths.push(
      path.join(__dirname, '../../backend/dist/cardioai-pro-backend.exe'),
      path.join(__dirname, '../../backend/dist/cardioai-pro-backend'),
      path.join(process.resourcesPath || '', 'backend', 'cardioai-pro-backend.exe'),
      path.join(process.resourcesPath || '', 'cardioai-pro-backend.exe'),
      path.join(__dirname, 'cardioai-pro-backend.exe')
    )
  } else {
    possiblePaths.push(
      path.join(__dirname, '../../backend/dist/cardioai-pro-backend'),
      path.join(__dirname, '../../backend/dist/cardioai-pro-backend.exe'),
      path.join(process.resourcesPath || '', 'backend', 'cardioai-pro-backend'),
      path.join(process.resourcesPath || '', 'cardioai-pro-backend'),
      path.join(__dirname, 'cardioai-pro-backend')
    )
  }
  
  backendPath = possiblePaths.find(p => {
    try {
      return require('fs').existsSync(p)
    } catch (error) {
      console.warn('Error checking path:', p, error)
      return false
    }
  })
  
  if (!backendPath) {
    console.error('Backend executable not found in any of these locations:')
    possiblePaths.forEach(p => console.error('  -', p))
    console.warn('Application will run in frontend-only mode')
    return
  }

  console.log('Starting backend server from:', backendPath)
  backendProcess = spawn(backendPath, [], {
    stdio: 'pipe',
    detached: false,
    windowsHide: true, // Hide console window on Windows
  })

  backendProcess.stdout.on('data', data => {
    console.log(`Backend: ${data}`)
  })

  backendProcess.stderr.on('data', data => {
    console.error(`Backend Error: ${data}`)
  })

  backendProcess.on('close', code => {
    console.log(`Backend process exited with code ${code}`)
  })
}

function stopBackend() {
  if (backendProcess) {
    console.log('Stopping backend server...')
    backendProcess.kill('SIGTERM')
    backendProcess = null
  }
}

app.whenReady().then(() => {
  startBackend()

  setTimeout(() => {
    createWindow()
  }, 2000)

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  stopBackend()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopBackend()
})

ipcMain.handle('get-backend-url', () => {
  return 'http://localhost:8000'
})

ipcMain.handle('check-backend-health', async () => {
  try {
    const response = await fetch('http://localhost:8000/health')
    return response.ok
  } catch (error) {
    return false
  }
})
