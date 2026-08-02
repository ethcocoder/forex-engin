import { autoUpdater } from "electron-updater"
import { log } from "./log"

export interface UpdateSettings {
  enabled: boolean
  feedUrl?: string
}

/**
 * Flag-gated auto-update. Electron-updater is only activated when the user
 * opts in with a generic feed URL in Settings (`updates.enabled` + feedUrl).
 * With no feed configured this is a silent no-op.
 */
export class Updater {
  private initialized = false

  apply(settings: UpdateSettings): void {
    if (this.initialized) return
    if (!settings.enabled || !settings.feedUrl) return

    try {
      autoUpdater.setFeedURL({ provider: "generic", url: settings.feedUrl })
      autoUpdater.autoDownload = true
      autoUpdater.autoInstallOnAppQuit = true

      autoUpdater.on("checking-for-update", () => log("updater: checking for update"))
      autoUpdater.on("update-available", (info) => log(`updater: update available (${info.version})`))
      autoUpdater.on("update-not-available", () => log("updater: no update available"))
      autoUpdater.on("error", (e) => log(`updater error: ${e instanceof Error ? e.message : String(e)}`, "error"))
      autoUpdater.on("update-downloaded", () => log("updater: update downloaded — will install on quit"))

      void autoUpdater.checkForUpdates()
      this.initialized = true
      log(`updater: enabled via ${settings.feedUrl}`)
    } catch (e) {
      log(`updater init failed: ${e instanceof Error ? e.message : String(e)}`, "error")
    }
  }
}
