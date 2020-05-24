/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "OnDiskInvertedLists.h"

//#include <pthread.h>

#include <algorithm>

#include <unordered_set>

//#include <unistd.h>

#ifdef _WIN32
#include "windows-mmap.h"
#else
#include <sys/mman.h>
#endif

#include <mutex>
#include <thread>

#include <sys/types.h>

#include "FaissAssert.h"

#ifdef _WIN32

#include <windows.h>

#undef min
#undef max

// Helper function taking a file name instead of a HANDLE
static bool truncate(const char* PathName, size_t NewSize) {
    HANDLE hFile = CreateFileA(PathName, GENERIC_WRITE, FILE_SHARE_READ,
        NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        return false;
    }
    LARGE_INTEGER Size;
    Size.QuadPart = NewSize;
    bool Success = (SetFilePointerEx(hFile, Size, NULL, FILE_BEGIN) &&
        SetEndOfFile(hFile));
    CloseHandle(hFile);
    return Success;
}

#endif

namespace faiss {


/**********************************************
 * LockLevels
 **********************************************/


struct LockLevels {
    /* There n times lock1(n), one lock2 and one lock3
     * Invariants:
     *    a single thread can hold one lock1(n) for some n
     *    a single thread can hold lock2, if it holds lock1(n) for some n
     *    a single thread can hold lock3, if it holds lock1(n) for some n
     *       AND lock2 AND no other thread holds lock1(m) for m != n
     */
    std::mutex mutex1;
    std::condition_variable level1_cv;
    std::condition_variable level2_cv;
    std::condition_variable level3_cv;

    std::unordered_set<int> level1_holders; // which level1 locks are held
    int n_level2; // nb threads that wait on level2
    bool level3_in_use; // a threads waits on level3
    bool level2_in_use;

    LockLevels() {
        //pthread_mutex_init(&mutex1, nullptr);
        //pthread_cond_init(&level1_cv, nullptr);
        //pthread_cond_init(&level2_cv, nullptr);
        //pthread_cond_init(&level3_cv, nullptr);
        n_level2 = 0;
        level2_in_use = false;
        level3_in_use = false;
    }

    ~LockLevels() {
        //pthread_cond_destroy(&level1_cv);
        //pthread_cond_destroy(&level2_cv);
        //pthread_cond_destroy(&level3_cv);
        //pthread_mutex_destroy(&mutex1);
    }

    void lock_1(int no) {
        std::unique_lock<std::mutex> locker(mutex1);
        while (level3_in_use || level1_holders.count(no) > 0) {
            level1_cv.wait(locker);
        }
        level1_holders.insert(no);
        //pthread_mutex_unlock(&mutex1);
    }

    void unlock_1(int no) {
        std::lock_guard<std::mutex> locker(mutex1);
        assert(level1_holders.count(no) == 1);
        level1_holders.erase(no);
        if (level3_in_use) { // a writer is waiting
            level3_cv.notify_one();
        } else {
            level1_cv.notify_all();
        }
        //pthread_mutex_unlock(&mutex1);
    }

    void lock_2() {
        std::unique_lock<std::mutex> locker(mutex1);
        n_level2 ++;
        if (level3_in_use) { // tell waiting level3 that we are blocked
            level3_cv.notify_one();
        }
        while (level2_in_use) {
            level2_cv.wait(locker);
        }
        level2_in_use = true;
        //pthread_mutex_unlock(&mutex1);
    }

    void unlock_2() {
        std::lock_guard<std::mutex> locker(mutex1);
        level2_in_use = false;
        n_level2 --;
        level2_cv.notify_one();
        //pthread_mutex_unlock(&mutex1);
    }

    void lock_3() {
        //pthread_mutex_lock(&mutex1);
        std::unique_lock<std::mutex> locker(mutex1);
        level3_in_use = true;
        // wait until there are no level1 holders anymore except the
        // ones that are waiting on level2 (we are holding lock2)
        while (level1_holders.size() > n_level2) {
            level3_cv.wait(locker);
        }
        // don't release the lock!
        locker.release();
    }

    void unlock_3() {
        level3_in_use = false;
        // wake up all level1_holders
        level1_cv.notify_all();
        //pthread_mutex_unlock(&mutex1);
        mutex1.unlock();
    }

    void print () {
        std::unique_lock<std::mutex> locker(mutex1);
        printf("State: level3_in_use=%d n_level2=%d level1_holders: [", level3_in_use, n_level2);
        for (int k : level1_holders) {
            printf("%d ", k);
        }
        printf("]\n");
        //pthread_mutex_unlock(&mutex1);
    }

};

/**********************************************
 * OngoingPrefetch
 **********************************************/

struct OnDiskInvertedLists::OngoingPrefetch {

    struct Thread {
        std::thread pth;
        const OnDiskInvertedLists *od;
        int64_t list_no;
    };

    std::vector<Thread> threads;

    std::mutex mutex;

    // pretext to avoid code below to be optimized out
    static int global_cs;

    const OnDiskInvertedLists *od;

    OngoingPrefetch (const OnDiskInvertedLists *od): od (od)
    {
        //pthread_mutex_init (&mutex, nullptr);
    }

    static void* prefetch_list (Thread *th) {
        //Thread *th = static_cast<Thread*>(arg);

        th->od->locks->lock_1(th->list_no);
        size_t n = th->od->list_size(th->list_no);
        const Index::idx_t *idx = th->od->get_ids(th->list_no);
        const uint8_t *codes = th->od->get_codes(th->list_no);
        int cs = 0;
        for (size_t i = 0; i < n;i++) {
            cs += idx[i];
        }
        const long *codes8 = (const long*)codes;
        long n8 = n * th->od->code_size / 8;

        for (size_t i = 0; i < n8;i++) {
            cs += codes8[i];
        }
        th->od->locks->unlock_1(th->list_no);
        global_cs += cs & 1;
        return nullptr;
    }

    void prefetch_lists (const int64_t *list_nos, int n) {
        std::unique_lock<std::mutex> locker(mutex);
        for (auto &th: threads) {
            if (th.list_no != -1) {
                //pthread_join (th.pth, nullptr);
                th.pth.join();
            }
        }
        threads.resize (n);
        for (int i = 0; i < n; i++) {
            long list_no = list_nos[i];
            Thread & th = threads[i];
            if (list_no >= 0 && od->list_size(list_no) > 0) {
                th.list_no = list_no;
                th.od = od;
                //pthread_create (&th.pth, nullptr, prefetch_list, &th);
                th.pth = std::thread(prefetch_list, &th);
            } else {
                th.list_no = -1;
            }
        }
        //pthread_mutex_unlock (&mutex);
    }

    ~OngoingPrefetch () {
        std::unique_lock<std::mutex> locker(mutex);
        for (auto &th: threads) {
            if (th.list_no != -1) {
                //pthread_join (th.pth, nullptr);
                th.pth.join();
            }
        }
        //pthread_mutex_unlock (&mutex);
        //pthread_mutex_destroy (&mutex);
    }

};

int OnDiskInvertedLists::OngoingPrefetch::global_cs = 0;


void OnDiskInvertedLists::prefetch_lists (const int64_t *list_nos, int n) const
{
    pf->prefetch_lists (list_nos, n);
}



/**********************************************
 * OnDiskInvertedLists: mmapping
 **********************************************/


void OnDiskInvertedLists::do_mmap ()
{
    const char *rw_flags = read_only ? "r" : "r+";
    int prot = read_only ? PROT_READ : PROT_WRITE | PROT_READ;
    FILE *f = fopen (filename.c_str(), rw_flags);
    FAISS_THROW_IF_NOT_FMT (f, "could not open %s in mode %s: %s",
                            filename.c_str(), rw_flags, strerror(errno));

    uint8_t * ptro = (uint8_t*)mmap (nullptr, totsize,
                          prot, MAP_SHARED, fileno (f), 0);

    FAISS_THROW_IF_NOT_FMT (ptro != MAP_FAILED,
                            "could not mmap %s: %s",
                            filename.c_str(),
                            strerror(errno));
    ptr = ptro;
    fclose (f);

}

void OnDiskInvertedLists::update_totsize (size_t new_size)
{

    // unmap file
    if (ptr != nullptr) {
        int err = munmap (ptr, totsize);
        FAISS_THROW_IF_NOT_FMT (err == 0, "mumap error: %s",
                                strerror(errno));
    }
    if (totsize == 0) {
        // must create file before truncating it
        FILE *f = fopen (filename.c_str(), "w");
        FAISS_THROW_IF_NOT_FMT (f, "could not open %s in mode W: %s",
                                filename.c_str(), strerror(errno));
        fclose (f);
    }

    if (new_size > totsize) {
        if (!slots.empty() &&
            slots.back().offset + slots.back().capacity == totsize) {
            slots.back().capacity += new_size - totsize;
        } else {
            slots.push_back (Slot(totsize, new_size - totsize));
        }
    } else {
        assert(!"not implemented");
    }

    totsize = new_size;

    // create file
    printf ("resizing %s to %ld bytes\n", filename.c_str(), totsize);

    int err = truncate (filename.c_str(), totsize);

    FAISS_THROW_IF_NOT_FMT (err == 0, "truncate %s to %ld: %s",
                            filename.c_str(), totsize,
                            strerror(errno));
    do_mmap ();
}






/**********************************************
 * OnDiskInvertedLists
 **********************************************/

#define INVALID_OFFSET (size_t)(-1)

OnDiskInvertedLists::List::List ():
    size (0), capacity (0), offset (INVALID_OFFSET)
{}

OnDiskInvertedLists::Slot::Slot (size_t offset, size_t capacity):
    offset (offset), capacity (capacity)
{}

OnDiskInvertedLists::Slot::Slot ():
    offset (0), capacity (0)
{}



OnDiskInvertedLists::OnDiskInvertedLists (
        size_t nlist, size_t code_size,
        const char *filename):
    InvertedLists (nlist, code_size),
    filename (filename),
    totsize (0),
    ptr (nullptr),
    read_only (false),
    locks (new LockLevels ()),
    pf (new OngoingPrefetch (this))
{
    lists.resize (nlist);

    // slots starts empty
}

OnDiskInvertedLists::OnDiskInvertedLists ():
    InvertedLists (0, 0),
    totsize (0),
    ptr (nullptr),
    read_only (false),
    locks (new LockLevels ()),
    pf (new OngoingPrefetch (this))
{
}

OnDiskInvertedLists::~OnDiskInvertedLists ()
{
    delete pf;

    // unmap all lists
    if (ptr != nullptr) {
        int err = munmap (ptr, totsize);
        FAISS_THROW_IF_NOT_FMT (err == 0,
                                "mumap error: %s",
                                strerror(errno));
    }
    delete locks;
}




size_t OnDiskInvertedLists::list_size(size_t list_no) const
{
    return lists[list_no].size;
}


const uint8_t * OnDiskInvertedLists::get_codes (size_t list_no) const
{
    return ptr + lists[list_no].offset;
}

const Index::idx_t * OnDiskInvertedLists::get_ids (size_t list_no) const
{
    return (const idx_t*)(ptr + lists[list_no].offset +
                          code_size * lists[list_no].capacity);
}


void OnDiskInvertedLists::update_entries (
      size_t list_no, size_t offset, size_t n_entry,
      const idx_t *ids_in, const uint8_t *codes_in)
{
    FAISS_THROW_IF_NOT (!read_only);
    if (n_entry == 0) return;
    const List & l = lists[list_no];
    assert (n_entry + offset <= l.size);
    idx_t *ids = const_cast<idx_t*>(get_ids (list_no));
    memcpy (ids + offset, ids_in, sizeof(ids_in[0]) * n_entry);
    uint8_t *codes = const_cast<uint8_t*>(get_codes (list_no));
    memcpy (codes + offset * code_size, codes_in, code_size * n_entry);
}

size_t OnDiskInvertedLists::add_entries (
           size_t list_no, size_t n_entry,
           const idx_t* ids, const uint8_t *code)
{
    FAISS_THROW_IF_NOT (!read_only);
    locks->lock_1 (list_no);
    size_t o = list_size (list_no);
    resize_locked (list_no, n_entry + o);
    update_entries (list_no, o, n_entry, ids, code);
    locks->unlock_1 (list_no);
    return o;
}

void OnDiskInvertedLists::resize (size_t list_no, size_t new_size)
{
    FAISS_THROW_IF_NOT (!read_only);
    locks->lock_1 (list_no);
    resize_locked (list_no, new_size);
    locks->unlock_1 (list_no);
}



void OnDiskInvertedLists::resize_locked (size_t list_no, size_t new_size)
{
    List & l = lists[list_no];

    if (new_size <= l.capacity &&
        new_size > l.capacity / 2) {
        l.size = new_size;
        return;
    }

    // otherwise we release the current slot, and find a new one

    locks->lock_2 ();
    free_slot (l.offset, l.capacity);

    List new_l;

    if (new_size == 0) {
        new_l = List();
    } else {
        new_l.size = new_size;
        new_l.capacity = 1;
        while (new_l.capacity < new_size) {
            new_l.capacity *= 2;
        }
        new_l.offset = allocate_slot (
            new_l.capacity * (sizeof(idx_t) + code_size));
    }

    // copy common data
    if (l.offset != new_l.offset) {
        size_t n = std::min (new_size, l.size);
        if (n > 0) {
            memcpy (ptr + new_l.offset, get_codes(list_no), n * code_size);
            memcpy (ptr + new_l.offset + new_l.capacity * code_size,
                    get_ids (list_no), n * sizeof(idx_t));
        }
    }

    lists[list_no] = new_l;
    locks->unlock_2 ();
}

size_t OnDiskInvertedLists::allocate_slot (size_t capacity) {
    // should hold lock2

    auto it = slots.begin();
    while (it != slots.end() && it->capacity < capacity) {
        it++;
    }

    if (it == slots.end()) {
        // not enough capacity
        size_t new_size = totsize == 0 ? 32 : totsize * 2;
        while (new_size - totsize < capacity)
            new_size *= 2;
        locks->lock_3 ();
        update_totsize(new_size);
        locks->unlock_3 ();
        it = slots.begin();
        while (it != slots.end() && it->capacity < capacity) {
            it++;
        }
        assert (it != slots.end());
    }

    size_t o = it->offset;
    if (it->capacity == capacity) {
        slots.erase (it);
    } else {
        // take from beginning of slot
        it->capacity -= capacity;
        it->offset += capacity;
    }

    return o;
}



void OnDiskInvertedLists::free_slot (size_t offset, size_t capacity) {

    // should hold lock2
    if (capacity == 0) return;

    auto it = slots.begin();
    while (it != slots.end() && it->offset <= offset) {
        it++;
    }

    size_t inf = 1ULL << 60;

    size_t end_prev = inf;
    if (it != slots.begin()) {
        auto prev = it;
        prev--;
        end_prev = prev->offset + prev->capacity;
    }

    size_t begin_next = 1LL << 60;
    if (it != slots.end()) {
        begin_next = it->offset;
    }

    assert (end_prev == inf || offset >= end_prev);
    assert (offset + capacity <= begin_next);

    if (offset == end_prev) {
        auto prev = it;
        prev--;
        if (offset + capacity == begin_next) {
            prev->capacity += capacity + it->capacity;
            slots.erase (it);
        } else {
            prev->capacity += capacity;
        }
    } else {
        if (offset + capacity == begin_next) {
            it->offset -= capacity;
            it->capacity += capacity;
        } else {
            slots.insert (it, Slot (offset, capacity));
        }
    }

    // TODO shrink global storage if needed
}


/*****************************************
 * Compact form
 *****************************************/

size_t OnDiskInvertedLists::merge_from (const InvertedLists **ils, int n_il)
{
    FAISS_THROW_IF_NOT_MSG (totsize == 0, "works only on an empty InvertedLists");

    std::vector<size_t> sizes (nlist);
    for (int i = 0; i < n_il; i++) {
        const InvertedLists *il = ils[i];
        FAISS_THROW_IF_NOT (il->nlist == nlist && il->code_size == code_size);

        for (size_t j = 0; j < nlist; j++)  {
            sizes [j] += il->list_size(j);
        }
    }

    size_t cums = 0;
    size_t ntotal = 0;
    for (size_t j = 0; j < nlist; j++)  {
        ntotal += sizes[j];
        lists[j].size = 0;
        lists[j].capacity = sizes[j];
        lists[j].offset = cums;
        cums += lists[j].capacity * (sizeof(idx_t) + code_size);
    }

    update_totsize (cums);

#pragma omp parallel for
    for (size_t j = 0; j < nlist; j++) {
        List & l = lists[j];
        for (int i = 0; i < n_il; i++) {
            const InvertedLists *il = ils[i];
            size_t n_entry = il->list_size(j);
            l.size += n_entry;
            update_entries (j, l.size - n_entry, n_entry,
                            il->get_ids(j),
                            il->get_codes(j));
        }
        assert (l.size == l.capacity);
    }

    return ntotal;
}






} // namespace faiss
