const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  getStatus: () => ipcRenderer.invoke('engine:status'),
  runChaos: () => ipcRenderer.invoke('engine:runChaos'),
})
