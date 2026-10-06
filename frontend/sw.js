const CACHE = "netra-ai-shell-v3";

const ASSETS = [
    "/",
    "/index.html",
    "/styles.css",
    "/app.js",
    "/manifest.json",
    "/eye-background.jpg"
];


// ========================================================
// INSTALL
// ========================================================

self.addEventListener("install", (event) => {

    event.waitUntil(

        caches.open(CACHE).then((cache) => {
            return cache.addAll(ASSETS);
        })

    );

    self.skipWaiting();
});


// ========================================================
// ACTIVATE
// ========================================================

self.addEventListener("activate", (event) => {

    event.waitUntil(

        caches.keys().then((cacheNames) => {

            return Promise.all(

                cacheNames

                    .filter((cacheName) => {
                        return cacheName !== CACHE;
                    })

                    .map((cacheName) => {
                        return caches.delete(cacheName);
                    })

            );

        })

    );

    self.clients.claim();
});


// ========================================================
// FETCH
// ========================================================

self.addEventListener("fetch", (event) => {

    const request = event.request;

    /*
     * For GET requests, try the network first.
     * This prevents old HTML, CSS and JavaScript
     * from being silently served from the cache.
     */

    if (request.method === "GET") {

        event.respondWith(

            fetch(request)

                .then((networkResponse) => {

                    /*
                     * Save a fresh copy for offline use.
                     */

                    const responseCopy =
                        networkResponse.clone();

                    caches.open(CACHE).then((cache) => {

                        cache.put(
                            request,
                            responseCopy
                        );

                    });

                    return networkResponse;
                })

                .catch(() => {

                    /*
                     * If the network is unavailable,
                     * fall back to the cached version.
                     */

                    return caches.match(request);

                })

        );

        return;
    }

    /*
     * POST requests, including /predict, are never
     * cached by the service worker.
     */

});