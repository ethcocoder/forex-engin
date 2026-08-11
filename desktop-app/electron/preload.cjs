const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  getStatus: () => ipcRenderer.invoke('engine:status'),
  saveConfig: (cfg) => ipcRenderer.invoke('engine:saveConfig', cfg),
  runChaos: () => ipcRenderer.invoke('engine:runChaos'),
})
